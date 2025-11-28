# coding: utf-8
"""
Chess game analysis tools for fetching games from chess.com and analyzing them.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import re
import chess
import chess.pgn
import chess.engine
from io import StringIO
import shutil


def find_stockfish():
    """
    Try to find Stockfish in common locations.
    Returns the path if found, None otherwise.
    """
    # Common paths to check
    common_paths = [
        'stockfish',  # In PATH
        '/usr/bin/stockfish',
        '/usr/local/bin/stockfish',
        '/usr/games/stockfish',
        '/opt/homebrew/bin/stockfish',  # macOS Homebrew
        'C:\\Program Files\\Stockfish\\stockfish.exe',  # Windows
        'C:\\Program Files (x86)\\Stockfish\\stockfish.exe',  # Windows 32-bit
    ]
    
    # First try if it's in PATH
    stockfish_path = shutil.which('stockfish')
    if stockfish_path:
        return stockfish_path
    
    # Try common paths
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return None


def calculate_material(board):
    """Calculate material count for both sides (only pawns, not other pieces)"""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    white_material = 0
    black_material = 0
    white_pawns = 0
    black_pawns = 0
    
    # Track piece types for material kind comparison
    white_piece_types = []
    black_piece_types = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            value = piece_values[piece_type]
            
            if piece.color == chess.WHITE:
                white_material += value
                if piece_type == chess.PAWN:
                    white_pawns += 1
                elif piece_type != chess.KING:  # Don't count king
                    white_piece_types.append(piece_type)
            else:
                black_material += value
                if piece_type == chess.PAWN:
                    black_pawns += 1
                elif piece_type != chess.KING:  # Don't count king
                    black_piece_types.append(piece_type)
    
    # Sort piece types for comparison
    white_piece_types.sort()
    black_piece_types.sort()
    
    # Check if material kinds are the same
    is_same_material_kind = (white_piece_types == black_piece_types and 
                             white_pawns == black_pawns)
    
    return {
        'white_material': white_material,
        'black_material': black_material,
        'material_balance': white_material - black_material,
        'white_pawns': white_pawns,
        'black_pawns': black_pawns,
        'is_same_material_kind': is_same_material_kind,
    }


def count_isolated_pawns(board, color):
    """Count isolated pawns for a given color"""
    isolated = 0
    files_with_pawns = set()
    
    # Find all pawns and their files
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            file = chess.square_file(square)
            files_with_pawns.add(file)
    
    # Check each pawn if it's isolated
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            file = chess.square_file(square)
            # Check adjacent files
            has_friendly_pawn = False
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file <= 7 and adj_file in files_with_pawns:
                    has_friendly_pawn = True
                    break
            if not has_friendly_pawn:
                isolated += 1
    
    return isolated


def count_doubled_pawns(board, color):
    """Count doubled pawns (multiple pawns on same file)"""
    file_counts = [0] * 8
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            file = chess.square_file(square)
            file_counts[file] += 1
    
    # Count files with 2+ pawns
    doubled = sum(1 for count in file_counts if count >= 2)
    return doubled


def count_passed_pawns(board, color):
    """Count passed pawns (no enemy pawns ahead on same or adjacent files)"""
    passed = 0
    enemy_color = not color
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Check if there are enemy pawns ahead
            has_enemy_pawn_ahead = False
            direction = 1 if color == chess.WHITE else -1
            
            for check_rank in range(rank + direction, 8 if color == chess.WHITE else -1, direction):
                for check_file in [file - 1, file, file + 1]:
                    if 0 <= check_file <= 7 and 0 <= check_rank <= 7:
                        check_square = chess.square(check_file, check_rank)
                        check_piece = board.piece_at(check_square)
                        if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == enemy_color:
                            has_enemy_pawn_ahead = True
                            break
                if has_enemy_pawn_ahead:
                    break
            
            if not has_enemy_pawn_ahead:
                passed += 1
    
    return passed


def get_king_safety(board, color):
    """Get king safety metrics"""
    king_square = board.king(color)
    if king_square is None:
        return {'castled': False, 'king_rank': None, 'king_file': None}
    
    king_rank = chess.square_rank(king_square)
    king_file = chess.square_file(king_square)
    
    # Check if castled (king on back rank and rooks moved)
    castled = False
    if color == chess.WHITE:
        castled = king_rank == 0 and king_file in [1, 2, 5, 6]  # Kingside or queenside
    else:
        castled = king_rank == 7 and king_file in [1, 2, 5, 6]
    
    return {
        'castled': castled,
        'king_rank': king_rank,
        'king_file': king_file
    }


def count_center_control(board, color):
    """Count pieces controlling center squares (d4, d5, e4, e5)"""
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    control_count = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            # Get all squares this piece attacks
            attacks = board.attacks(square)
            if any(center in attacks for center in center_squares):
                control_count += 1
    
    return control_count


def get_game_phase(full_move, total_pieces):
    """
    Determine game phase based on move number and piece count.
    
    Parameters:
    -----------
    full_move : int
        Full move number (1 = White's 1st move, 2 = Black's 1st move, etc.)
    total_pieces : int
        Total number of pieces on the board (including kings)
    
    Returns:
    --------
    str: 'opening', 'middlegame', or 'endgame'
    """
    # Opening: first 10 moves
    if full_move <= 10:
        return 'opening'
    
    # Endgame: 16 or fewer pieces
    if total_pieces <= 16:
        return 'endgame'
    
    # Everything else is middlegame
    return 'middlegame'


def classify_move(eval_change):
    """
    Classify a move based on evaluation change from the player's perspective.
    
    Parameters:
    -----------
    eval_change : float
        Evaluation change in pawns from the player's perspective
        (positive = good for player, negative = bad for player)
    
    Returns:
    --------
    str: Move classification: 'blunder', 'mistake', 'inaccuracy', 'normal', 'good'
    """
    if eval_change <= -2.0:
        return 'blunder'
    elif eval_change <= -1.0:
        return 'mistake'
    elif eval_change <= -0.5:
        return 'inaccuracy'
    elif eval_change >= 0.5:
        return 'good'
    else:
        return 'normal'


def parse_time_control(time_control_str):
    """
    Parse TimeControl string (e.g., "180+2" = 180 seconds + 2 second increment)
    
    Returns:
    --------
    tuple: (initial_time_seconds, increment_seconds)
    """
    if not time_control_str:
        return None, None
    
    # Handle formats like "180+2", "600+0", etc.
    if '+' in time_control_str:
        parts = time_control_str.split('+')
        initial = int(parts[0])
        increment = int(parts[1]) if len(parts) > 1 else 0
        return initial, increment
    
    # Handle other formats if needed
    try:
        initial = int(time_control_str)
        return initial, 0
    except:
        return None, None


def parse_clock_time(clock_str):
    """
    Parse clock time string (e.g., "0:03:02.1" or "3:02.1")
    
    Returns:
    --------
    float: Time in seconds, or None if parsing fails
    """
    if not clock_str:
        return None
    
    try:
        # Remove [%clk and ] if present
        clock_str = clock_str.replace('[%clk', '').replace(']', '').strip()
        
        # Parse format: H:MM:SS.m or M:SS.m
        parts = clock_str.split(':')
        
        if len(parts) == 3:
            # Format: H:MM:SS.m
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            # Format: M:SS.m
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return None
    except:
        return None


def extract_clock_from_comment(comment):
    """
    Extract clock time from PGN comment (e.g., "{[%clk 0:03:02.1]}")
    
    Returns:
    --------
    float: Time in seconds, or None if not found
    """
    if not comment:
        return None
    
    # Look for [%clk pattern
    match = re.search(r'\[%clk\s+([^\]]+)\]', comment)
    if match:
        return parse_clock_time(match.group(1))
    return None


def fetch_chess_com_games(username, date, save_dir='chess_games', is_verbose=True):
    """
    Fetch chess games from chess.com API for a specific user and date.
    Saves games as PGN files and returns a DataFrame with game metadata.
    
    Parameters:
    -----------
    username : str
        Chess.com username
    date : str or datetime
        Date in format 'YYYY-MM-DD' or datetime object
    save_dir : str
        Directory to save PGN files (default: 'chess_games')
    is_verbose : bool
        If True, print detailed progress information (default: True)
    
    Returns:
    --------
    pd.DataFrame: DataFrame with game metadata including unique game_id and pgn_path
    """
    # Convert date to datetime if string
    if isinstance(date, str):
        date_obj = datetime.strptime(date, '%Y-%m-%d')
    else:
        date_obj = date
    
    year = date_obj.year
    month = date_obj.month
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Chess.com API JSON endpoint (provides metadata)
    json_url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month:02d}"
    
    if is_verbose:
        print(f"Fetching games for {username} from {year}-{month:02d}...")
        print(f"URL: {json_url}")
    
    # Chess.com API requires User-Agent header to avoid 403 errors
    headers = {
        'User-Agent': 'ChessGameAnalyzer/1.0 (https://github.com/yourusername/chess-analyzer)'
    }
    
    try:
        # Fetch JSON metadata
        response = requests.get(json_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        games_list = data.get('games', [])
        
        if not games_list:
            if is_verbose:
                print(f"No games found for {username} in {year}-{month:02d}")
            return pd.DataFrame()
        
        if is_verbose:
            print(f"Found {len(games_list)} games")
        
        # Extract metadata and save PGN files
        games_data = []
        
        for i, game in enumerate(games_list):
            # Extract unique game ID from URL
            game_url = game.get('url', '')
            # Extract game ID from URL (e.g., "https://www.chess.com/game/live/145059983558" -> "145059983558")
            game_id_match = re.search(r'/(\d+)$', game_url)
            if game_id_match:
                unique_game_id = game_id_match.group(1)
            else:
                # Fallback: use index if can't extract from URL
                unique_game_id = f"{username}_{year}{month:02d}_{i+1:04d}"
            
            # Get PGN content
            pgn_content = game.get('pgn', '')
            
            # Save PGN file
            filename = f"{username}_{year}-{month:02d}_game_{unique_game_id}.pgn"
            filepath = save_path / filename
            
            if pgn_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(pgn_content)
            
            # Extract game metadata
            white = game.get('white', {})
            black = game.get('black', {})
            
            # Calculate points based on result
            # win = 1, draw = 0.5, loss = 0
            white_result = white.get('result', '')
            black_result = black.get('result', '')
            
            white_points = 0.0
            if white_result == 'win':
                white_points = 1.0
            elif white_result == 'draw':
                white_points = 0.5
            
            black_points = 0.0
            if black_result == 'win':
                black_points = 1.0
            elif black_result == 'draw':
                black_points = 0.5
            
            # Extract time information
            end_time = game.get('end_time', 0)
            end_date = datetime.fromtimestamp(end_time) if end_time else None
            
            # Build game record
            game_record = {
                'game_id': unique_game_id,  # Unique identifier
                'pgn_path': str(filepath),  # Path to PGN file
                'url': game_url,
                'white_username': white.get('username', ''),
                'white_rating': white.get('rating', None),
                'white_result': white_result,
                'white_points': white_points,
                'black_username': black.get('username', ''),
                'black_rating': black.get('rating', None),
                'black_result': black_result,
                'black_points': black_points,
                'time_control': game.get('time_control', ''),
                'time_class': game.get('time_class', ''),  # blitz, rapid, bullet, etc.
                'rules': game.get('rules', ''),  # chess, chess960, etc.
                'end_time': end_date,
                'end_timestamp': end_time,
                'rated': game.get('rated', False),
            }
            
            games_data.append(game_record)
            if is_verbose:
                print(f"  [{i+1}/{len(games_list)}] Saved: {filename} (ID: {unique_game_id})")
        
        # Create DataFrame
        df = pd.DataFrame(games_data)
        
        # Convert end_time to datetime if available
        if 'end_time' in df.columns:
            df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        
        # Save DataFrame to CSV
        csv_filename = f"{username}_{year}-{month:02d}_games.csv"
        csv_filepath = save_path / csv_filename
        df.to_csv(csv_filepath, index=False, encoding='utf-8')
        
        if is_verbose:
            print(f"\n✅ Successfully saved {len(df)} games to {save_dir}/")
            print(f"📊 DataFrame created with {len(df)} rows and {len(df.columns)} columns")
            print(f"💾 Saved metadata to CSV: {csv_filename}")
        
        return df
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            if is_verbose:
                print(f"❌ No games found for user '{username}' in {year}-{month:02d}")
        elif e.response.status_code == 403:
            if is_verbose:
                print(f"❌ Access forbidden. Chess.com may be blocking the request.")
                print(f"   Try checking if the username is correct and the date is valid.")
        else:
            if is_verbose:
                print(f"❌ HTTP Error {e.response.status_code}: {e}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        if is_verbose:
            print(f"❌ Request Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        if is_verbose:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        return pd.DataFrame()


def analyze_pgn_evaluations(pgn, engine_path=None, depth=10, data_dir=None):
    """
    Analyze a PGN game and return a DataFrame with move numbers, engine evaluations, and position metrics.
    
    Parameters:
    -----------
    pgn : str or Path
        Path to PGN file or PGN text content
    engine_path : str, optional
        Path to chess engine executable. If None, will try to find Stockfish automatically.
        Common paths: 'stockfish', '/usr/bin/stockfish', '/usr/local/bin/stockfish'
    depth : int
        Engine search depth (default: 10)
    data_dir : str or Path, optional
        Path to folder with PGN and CSV files. If provided and CSV file with analysis exists,
        will load from disk instead of analyzing. If not provided or CSV doesn't exist, will analyze and save.
    
    Returns:
    --------
    pd.DataFrame: DataFrame with move-by-move analysis
    """
    # Read PGN from file or use as text
    if isinstance(pgn, (str, Path)):
        pgn_path = Path(pgn)
        if pgn_path.exists():
            # It's a file path
            pgn_file_path = pgn_path
            with open(pgn_path, 'r', encoding='utf-8') as f:
                pgn_text = f.read()
        else:
            # It's PGN text content
            pgn_text = str(pgn)
            pgn_file_path = None
    else:
        pgn_text = str(pgn)
        pgn_file_path = None
    
    # Check if we should load from disk
    if pgn_file_path is not None:
        # Determine where to look for CSV file
        if data_dir is not None:
            search_dir = Path(data_dir)
        else:
            # Look in same directory as PGN file
            search_dir = pgn_file_path.parent
        
        # Try to find corresponding CSV file
        # CSV filename should match PGN filename but with _analysis.csv extension
        pgn_stem = pgn_file_path.stem
        csv_file_path = search_dir / f"{pgn_stem}_analysis.csv"
        
        if csv_file_path.exists():
            # Load from CSV
            try:
                df = pd.read_csv(csv_file_path, encoding='utf-8')
                # Convert datetime columns if they exist
                for col in df.columns:
                    if 'time' in col.lower() and df[col].dtype == 'object':
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                return df
            except Exception as e:
                # If loading fails, continue with analysis
                pass
    
    # Parse PGN
    game = chess.pgn.read_game(StringIO(pgn_text))
    if not game:
        raise ValueError("Could not parse PGN game")
    
    # Get all moves
    board = game.board()
    moves = list(game.mainline_moves())
    
    if not moves:
        return pd.DataFrame(columns=['move_number', 'full_move', 'player', 'move_san', 
                                     'evaluation', 'evaluation_pawns'])
    
    # Parse time control from headers
    time_control_str = game.headers.get('TimeControl', '')
    initial_time, increment = parse_time_control(time_control_str)
    
    # Extract clock times from game nodes
    clock_times = []
    node = game
    move_index = 0
    
    while node.variations:
        node = node.variation(0)
        if move_index < len(moves):
            comment = node.comment
            clock_time = extract_clock_from_comment(comment)
            clock_times.append(clock_time)
            move_index += 1
        else:
            break
    
    # Initialize time tracking
    white_time_remaining = initial_time if initial_time else None
    black_time_remaining = initial_time if initial_time else None
    white_time_used = 0.0
    black_time_used = 0.0
    
    # Find engine if not specified
    if engine_path is None:
        engine_path = find_stockfish()
        if engine_path is None:
            raise ValueError(
                "Stockfish not found. Please install Stockfish or specify the engine_path.\n"
                "Installation:\n"
                "  Ubuntu/Debian: sudo apt-get install stockfish\n"
                "  macOS: brew install stockfish\n"
                "  Or download from: https://stockfishchess.org/download/\n"
                "  Then specify the path: analyze_pgn_evaluations(pgn, engine_path='/path/to/stockfish')"
            )
    
    # Initialize engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as e:
        raise ValueError(
            f"Could not start chess engine at '{engine_path}'.\n"
            f"Error: {e}\n\n"
            f"Please make sure:\n"
            f"  1. Stockfish is installed\n"
            f"  2. The path is correct\n"
            f"  3. The file has execute permissions\n\n"
            f"Installation:\n"
            f"  Ubuntu/Debian: sudo apt-get install stockfish\n"
            f"  macOS: brew install stockfish\n"
            f"  Or download from: https://stockfishchess.org/download/"
        )
    
    evaluations_data = []
    
    # Track cumulative exchanges (captures)
    white_captures = 0
    black_captures = 0
    white_promotions = 0
    black_promotions = 0
    
    try:
        # Evaluate initial position
        board = game.board()
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        eval_score = info['score'].relative.score(mate_score=10000)
        if eval_score is None:
            eval_score = 0
        
        # Calculate initial metrics
        material = calculate_material(board)
        white_king_safety = get_king_safety(board, chess.WHITE)
        black_king_safety = get_king_safety(board, chess.BLACK)
        
        # Add initial position (move 0)
        eval_pawns = eval_score / 100.0 if eval_score else 0.0
        total_pieces = len(board.piece_map())
        game_phase = get_game_phase(0, total_pieces)
        
        initial_data = {
            'move_number': 0,
            'full_move': 0,
            'player': 'Start',
            'move_san': '',
            'evaluation': eval_score,
            'evaluation_pawns': eval_pawns,
            'eval_change': 0.0,  # No change for initial position
            'move_type': 'normal',  # No move classification for start
            'game_phase': game_phase,
            'white_time_remaining': white_time_remaining,
            'black_time_remaining': black_time_remaining,
            'white_time_used': 0.0,
            'black_time_used': 0.0,
            'time_used': None,  # No move at start
            **material,
            'white_isolated_pawns': count_isolated_pawns(board, chess.WHITE),
            'black_isolated_pawns': count_isolated_pawns(board, chess.BLACK),
            'white_doubled_pawns': count_doubled_pawns(board, chess.WHITE),
            'black_doubled_pawns': count_doubled_pawns(board, chess.BLACK),
            'white_passed_pawns': count_passed_pawns(board, chess.WHITE),
            'black_passed_pawns': count_passed_pawns(board, chess.BLACK),
            'white_center_control': count_center_control(board, chess.WHITE),
            'black_center_control': count_center_control(board, chess.BLACK),
            'white_castled': white_king_safety['castled'],
            'black_castled': black_king_safety['castled'],
            'white_captures': white_captures,
            'black_captures': black_captures,
            'white_promotions': white_promotions,
            'black_promotions': black_promotions,
            'is_capture': False,
            'is_promotion': False,
        }
        evaluations_data.append(initial_data)
        
        # Store previous evaluation for calculating change
        prev_eval_pawns = eval_pawns
        
        # Evaluate after each move
        for i, move in enumerate(moves):
            # Check if this is a capture before pushing
            is_capture = board.is_capture(move)
            is_promotion = bool(move.promotion)
            
            # Get move in SAN notation before pushing
            move_san = board.san(move)
            
            # Track captures and promotions
            player = 'White' if i % 2 == 0 else 'Black'
            if is_capture:
                if player == 'White':
                    white_captures += 1
                else:
                    black_captures += 1
            
            if is_promotion:
                if player == 'White':
                    white_promotions += 1
                else:
                    black_promotions += 1
            
            # Push move to board
            board.push(move)
            
            # Get evaluation (this is relative to the side to move)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            eval_score = info['score'].relative.score(mate_score=10000)
            if eval_score is None:
                eval_score = 0
            
            # Convert evaluation to always be relative to White
            # If it's Black's turn, flip the sign
            if board.turn == chess.BLACK:
                eval_score = -eval_score
            
            # Determine player
            full_move = (i // 2) + 1
            
            # Calculate evaluation change from the player's perspective
            eval_pawns = eval_score / 100.0 if eval_score else 0.0
            
            # Calculate change: from White's perspective
            eval_change_white_perspective = eval_pawns - prev_eval_pawns
            
            # Convert to the player's perspective
            # If White moved: positive change = good, negative = bad
            # If Black moved: negative change (from White's perspective) = good for Black
            if player == 'White':
                eval_change = eval_change_white_perspective
            else:  # Black
                eval_change = -eval_change_white_perspective
            
            # Classify the move
            move_type = classify_move(eval_change)
            
            # Calculate time used and remaining
            time_used = None
            if i < len(clock_times) and clock_times[i] is not None:
                current_clock = clock_times[i]
                
                if player == 'White':
                    if white_time_remaining is not None:
                        # Time used = previous time + increment - current time
                        prev_time = white_time_remaining
                        time_used = prev_time + (increment if increment else 0) - current_clock
                        if time_used < 0:
                            time_used = 0  # Can't be negative
                        white_time_used += time_used
                        white_time_remaining = current_clock
                else:  # Black
                    if black_time_remaining is not None:
                        # Time used = previous time + increment - current time
                        prev_time = black_time_remaining
                        time_used = prev_time + (increment if increment else 0) - current_clock
                        if time_used < 0:
                            time_used = 0  # Can't be negative
                        black_time_used += time_used
                        black_time_remaining = current_clock
            
            # Calculate all metrics
            material = calculate_material(board)
            white_king_safety = get_king_safety(board, chess.WHITE)
            black_king_safety = get_king_safety(board, chess.BLACK)
            
            # Determine game phase
            total_pieces = len(board.piece_map())
            game_phase = get_game_phase(full_move, total_pieces)
            
            move_data = {
                'move_number': i + 1,
                'full_move': full_move,
                'player': player,
                'move_san': move_san,
                'evaluation': eval_score,
                'evaluation_pawns': eval_pawns,
                'eval_change': eval_change,
                'move_type': move_type,
                'game_phase': game_phase,
                'white_time_remaining': white_time_remaining,
                'black_time_remaining': black_time_remaining,
                'white_time_used': white_time_used,
                'black_time_used': black_time_used,
                'time_used': time_used,  # Time used for this specific move
                **material,
                'white_isolated_pawns': count_isolated_pawns(board, chess.WHITE),
                'black_isolated_pawns': count_isolated_pawns(board, chess.BLACK),
                'white_doubled_pawns': count_doubled_pawns(board, chess.WHITE),
                'black_doubled_pawns': count_doubled_pawns(board, chess.BLACK),
                'white_passed_pawns': count_passed_pawns(board, chess.WHITE),
                'black_passed_pawns': count_passed_pawns(board, chess.BLACK),
                'white_center_control': count_center_control(board, chess.WHITE),
                'black_center_control': count_center_control(board, chess.BLACK),
                'white_castled': white_king_safety['castled'],
                'black_castled': black_king_safety['castled'],
                'white_captures': white_captures,
                'black_captures': black_captures,
                'white_promotions': white_promotions,
                'black_promotions': black_promotions,
                'is_capture': is_capture,
                'is_promotion': is_promotion,
            }
            
            evaluations_data.append(move_data)
            
            # Update previous evaluation for next iteration
            prev_eval_pawns = eval_pawns
    
    finally:
        engine.quit()
    
    # Create DataFrame
    df = pd.DataFrame(evaluations_data)
    
    # Save to CSV if data_dir is provided and we have a PGN file path
    if data_dir is not None and pgn_file_path is not None:
        data_dir_path = Path(data_dir)
        data_dir_path.mkdir(exist_ok=True)
        pgn_stem = pgn_file_path.stem
        csv_file_path = data_dir_path / f"{pgn_stem}_analysis.csv"
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
    elif pgn_file_path is not None:
        # If no data_dir specified but we have a file path, save in same directory as PGN
        pgn_dir = pgn_file_path.parent
        pgn_stem = pgn_file_path.stem
        csv_file_path = pgn_dir / f"{pgn_stem}_analysis.csv"
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
    
    return df


def analyze_games_from_chess_com(username, date, save_dir='chess_games', engine_path=None, depth=10, is_verbose=True, data_dir=None):
    """
    Fetch games from chess.com and analyze all of them, returning a wide DataFrame with all moves from all games.
    
    Parameters:
    -----------
    username : str
        Chess.com username
    date : str or datetime
        Date in format 'YYYY-MM-DD' or datetime object
    save_dir : str
        Directory to save PGN files (default: 'chess_games')
    engine_path : str, optional
        Path to chess engine executable. If None, will try to find Stockfish automatically.
    depth : int
        Engine search depth (default: 10)
    is_verbose : bool
        If True, print detailed progress information (default: True)
    data_dir : str or Path, optional
        Path to folder with PGN and CSV files. If provided, will try to load analysis from CSV files
        instead of re-analyzing. If not provided, uses save_dir as data_dir.
    
    Returns:
    --------
    pd.DataFrame: Wide DataFrame with all moves from all games, including game metadata
    """
    # Use save_dir as data_dir if data_dir not provided
    if data_dir is None:
        data_dir = save_dir
    
    # Step 1: Fetch games
    if is_verbose:
        print("=" * 80)
        print("STEP 1: FETCHING GAMES FROM CHESS.COM")
        print("=" * 80)
    
    df_games = fetch_chess_com_games(username, date, save_dir=save_dir, is_verbose=is_verbose)
    
    if df_games.empty:
        if is_verbose:
            print("No games found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Step 2: Analyze each game
    if is_verbose:
        print("\n" + "=" * 80)
        print("STEP 2: ANALYZING GAMES")
        print("=" * 80)
    
    all_moves_data = []
    
    for idx, game_row in df_games.iterrows():
        game_id = game_row['game_id']
        pgn_path = game_row['pgn_path']
        
        if is_verbose:
            print(f"\nAnalyzing game {idx + 1}/{len(df_games)}: {game_id}")
        
        try:
            # Analyze the game (will load from CSV if available)
            df_moves = analyze_pgn_evaluations(pgn_path, engine_path=engine_path, depth=depth, data_dir=data_dir)
            
            if not df_moves.empty:
                # Add game metadata to each move row
                for col in df_games.columns:
                    if col != 'pgn_path':  # Don't duplicate pgn_path
                        df_moves[col] = game_row[col]
                
                all_moves_data.append(df_moves)
                
                if is_verbose:
                    print(f"  ✅ Analyzed {len(df_moves)} moves")
            else:
                if is_verbose:
                    print(f"  ⚠️  No moves found in game")
        
        except Exception as e:
            if is_verbose:
                print(f"  ❌ Error analyzing game {game_id}: {e}")
            continue
    
    # Step 3: Combine all moves into one wide DataFrame
    if not all_moves_data:
        if is_verbose:
            print("\nNo games were successfully analyzed.")
        return pd.DataFrame()
    
    if is_verbose:
        print("\n" + "=" * 80)
        print("STEP 3: COMBINING ALL MOVES")
        print("=" * 80)
    
    df_all_moves = pd.concat(all_moves_data, ignore_index=True)
    
    if is_verbose:
        print(f"✅ Created wide DataFrame with {len(df_all_moves)} moves from {len(all_moves_data)} games")
        print(f"📊 DataFrame shape: {df_all_moves.shape}")
        print(f"📋 Columns: {len(df_all_moves.columns)}")
    
    return df_all_moves

