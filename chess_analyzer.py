# coding: utf-8
"""
Chess game analysis tools for fetching games from chess.com and analyzing them.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import re
import chess
import chess.pgn
import chess.engine
from io import StringIO
import shutil
import os

# Default data directory - absolute path
DEFAULT_DATA_DIR = Path(__file__).parent / 'data'


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


def fetch_chess_com_games(username, date, save_dir=None, is_verbose=True):
    """
    Fetch chess games from chess.com API for a specific user and date.
    Saves games as PGN files to data/pgn and metadata to data/game_meta.
    Returns a DataFrame with game metadata.
    
    Parameters:
    -----------
    username : str
        Chess.com username
    date : str or datetime
        Date in format 'YYYY-MM-DD' or datetime object
    save_dir : str or Path, optional
        Base directory (default: DEFAULT_DATA_DIR). Files saved to save_dir/pgn and save_dir/game_meta
    is_verbose : bool
        If True, print detailed progress information (default: True)
    
    Returns:
    --------
    pd.DataFrame: DataFrame with game metadata including unique game_id and pgn_path
    """
    # Use default data directory if not specified
    if save_dir is None:
        save_dir = DEFAULT_DATA_DIR
    else:
        save_dir = Path(save_dir)
    
    # Convert date to datetime if string
    if isinstance(date, str):
        date_obj = datetime.strptime(date, '%Y-%m-%d')
    else:
        date_obj = date
    
    year = date_obj.year
    month = date_obj.month
    
    # Create save directories using data/ structure
    base_dir = Path(save_dir)
    pgn_dir = base_dir / 'pgn'
    game_meta_dir = base_dir / 'game_meta'
    pgn_dir.mkdir(parents=True, exist_ok=True)
    game_meta_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            # Extract game date from PGN content or use end_time
            game_date_str = date_obj.strftime('%Y-%m-%d')  # Default to the date parameter
            if pgn_content:
                # Try to extract date from PGN headers
                date_match = re.search(r'\[Date\s+"([^"]+)"\]', pgn_content)
                if date_match:
                    pgn_date_str = date_match.group(1)
                    # Convert from PGN format (YYYY.MM.DD) to our format (YYYY-MM-DD)
                    try:
                        pgn_date = datetime.strptime(pgn_date_str, '%Y.%m.%d')
                        game_date_str = pgn_date.strftime('%Y-%m-%d')
                    except:
                        pass  # Use default if parsing fails
            
            # Save PGN file to pgn folder with pattern: yyyy-mm-dd_game_id.pgn
            filename = f"{game_date_str}_game_{unique_game_id}.pgn"
            filepath = pgn_dir / filename
            
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
                'game_date': game_date_str,
            }
            
            games_data.append(game_record)
            
            # Save metadata to data/game_meta folder
            meta_filename = f"{game_date_str}_game_{unique_game_id}.csv"
            meta_filepath = game_meta_dir / meta_filename
            meta_df = pd.DataFrame([game_record])
            meta_df.to_csv(meta_filepath, index=False, encoding='utf-8')
            
            if is_verbose:
                print(f"  [{i+1}/{len(games_list)}] Saved: {filename} (ID: {unique_game_id})")
        
        # Create DataFrame
        df = pd.DataFrame(games_data)
        
        # Convert end_time to datetime if available
        if 'end_time' in df.columns:
            df['end_time'] = pd.to_datetime(df['end_time'])
        
        if is_verbose:
            print(f"\n✅ Successfully saved {len(df)} games to {save_dir}/")
            print(f"📊 DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        
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
    Saves results to data/games_analysis folder.
    
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
        Base directory (default: DEFAULT_DATA_DIR). Analysis saved to data_dir/games_analysis.
        If CSV file with analysis exists, will load from disk instead of analyzing.
    
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
    
    # Use default data directory if not specified
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)
    
    # Check if we should load from disk
    if pgn_file_path is not None:
        # Try to extract game date and ID from filename
        game_date_str = None
        game_id = None
        
        # Extract date from filename (pattern: yyyy-mm-dd_game_id.pgn)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', pgn_file_path.stem)
        if date_match:
            game_date_str = date_match.group(1)
        
        # Extract game_id from filename
        game_id_match = re.search(r'game_(\d+)', pgn_file_path.stem)
        if game_id_match:
            game_id = game_id_match.group(1)
        
        # Try to load from data/games_analysis
        if game_date_str and game_id:
            data_dir_path = Path(data_dir)
            games_analysis_dir = data_dir_path / 'games_analysis'
            csv_file_path = games_analysis_dir / f"{game_date_str}_game_{game_id}.csv"
            
            if csv_file_path.exists():
                # Load from CSV
                try:
                    df = pd.read_csv(csv_file_path, encoding='utf-8')
                    # Convert datetime columns if they exist
                    for col in df.columns:
                        if 'time' in col.lower() and df[col].dtype == 'object':
                            try:
                                df[col] = pd.to_datetime(df[col])
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
    
    # Extract game date and ID from PGN headers or filename
    game_date_str = None
    game_id = None
    
    # Try to get date from PGN headers
    date_header = game.headers.get('Date', '')
    if date_header:
        try:
            # PGN date format is YYYY.MM.DD
            pgn_date = datetime.strptime(date_header, '%Y.%m.%d')
            game_date_str = pgn_date.strftime('%Y-%m-%d')
        except:
            pass
    
    # Try to extract game_id from filename if available
    if pgn_file_path is not None:
        game_id_match = re.search(r'game_(\d+)', pgn_file_path.stem)
        if game_id_match:
            game_id = game_id_match.group(1)
        # If date not found in headers, try to extract from filename
        if not game_date_str:
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', pgn_file_path.stem)
            if date_match:
                game_date_str = date_match.group(1)
    
    # Get all moves
    board = game.board()
    moves = list(game.mainline_moves())
    
    if not moves:
        return pd.DataFrame(columns=['move_number', 'full_move', 'player', 'move_san', 
                                     'evaluation', 'evaluation_pawns'])
    
    # Parse time control from headers
    time_control_str = game.headers.get('TimeControl', '')
    initial_time, increment = parse_time_control(time_control_str)
    
    # Extract ECO code from headers
    eco_code = game.headers.get('ECO', '')
    
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
            'eco_code': eco_code,  # ECO code of the opening
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
                'eco_code': eco_code,  # ECO code of the opening
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
    
    # Save to CSV in data/games_analysis folder with pattern: yyyy-mm-dd_game_id.csv
    if game_date_str and game_id:
        data_dir_path = Path(data_dir)
        games_analysis_dir = data_dir_path / 'games_analysis'
        games_analysis_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = f"{game_date_str}_game_{game_id}.csv"
        csv_file_path = games_analysis_dir / csv_filename
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
    else:
        # Fallback if date/id not available - save to data/games_analysis anyway
        data_dir_path = Path(data_dir)
        games_analysis_dir = data_dir_path / 'games_analysis'
        games_analysis_dir.mkdir(parents=True, exist_ok=True)
        if pgn_file_path is not None:
            pgn_stem = pgn_file_path.stem
            csv_file_path = games_analysis_dir / f"{pgn_stem}_analysis.csv"
        else:
            csv_file_path = games_analysis_dir / "unknown_game_analysis.csv"
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
    
    return df


def analyze_games_from_chess_com(username, date, save_dir=None, engine_path=None, depth=10, is_verbose=True, data_dir=None):
    """
    Fetch games from chess.com and analyze all of them, returning a wide DataFrame with all moves from all games.
    Saves PGN files to data/pgn, metadata to data/game_meta, and analysis to data/games_analysis.
    
    Parameters:
    -----------
    username : str
        Chess.com username
    date : str or datetime
        Date in format 'YYYY-MM-DD' or datetime object
    save_dir : str or Path, optional
        Base directory (default: DEFAULT_DATA_DIR). Files saved to save_dir/pgn, save_dir/game_meta, save_dir/games_analysis
    engine_path : str, optional
        Path to chess engine executable. If None, will try to find Stockfish automatically.
    depth : int
        Engine search depth (default: 10)
    is_verbose : bool
        If True, print detailed progress information (default: True)
    data_dir : str or Path, optional
        Base directory for loading analysis CSV files (default: same as save_dir).
        Analysis files are in data_dir/games_analysis.
    
    Returns:
    --------
    pd.DataFrame: Wide DataFrame with all moves from all games, including game metadata
    """
    # Use default data directory if not specified
    if save_dir is None:
        save_dir = DEFAULT_DATA_DIR
    else:
        save_dir = Path(save_dir)
    
    # Use save_dir as data_dir if data_dir not provided
    if data_dir is None:
        data_dir = save_dir
    else:
        data_dir = Path(data_dir)
    
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
            # Analyze the game (will load from CSV in data/games_analysis if available)
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


def start_analyze(user_name, date_range, is_verbose=True, is_api=1, path=None, engine_path=None, depth=10):
    """
    Analyze chess games from chess.com for a user over a date range.
    This is a convenience function that handles date ranges and combines results.
    
    Parameters:
    -----------
    user_name : str
        Chess.com username
    date_range : tuple
        Tuple of (start_date, end_date) in format 'YYYY-MM-DD'
    is_verbose : bool
        If True, print detailed progress information (default: True)
    is_api : int
        If 1, fetch games from chess.com API. If 0, only analyze existing PGN files (default: 1)
    path : str or Path, optional
        Base directory (default: DEFAULT_DATA_DIR). Files saved to path/pgn, path/game_meta, path/games_analysis
    engine_path : str, optional
        Path to chess engine executable. If None, will try to find Stockfish automatically.
    depth : int
        Engine search depth (default: 10)
    
    Returns:
    --------
    pd.DataFrame: Wide DataFrame with all moves from all games, including game metadata and ECO codes
    """
    # Use default data directory if not specified
    if path is None:
        path = DEFAULT_DATA_DIR
    else:
        path = Path(path)
    
    start_date_str, end_date_str = date_range
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    all_dataframes = []
    
    if is_api == 1:
        # Fetch and analyze games from chess.com API (day by day)
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            if is_verbose:
                print(f"\n{'=' * 80}")
                print(f"Processing date: {date_str}")
                print(f"{'=' * 80}")
            
            # Fetch and analyze games from chess.com API
            df_date = analyze_games_from_chess_com(
                username=user_name,
                date=current_date,
                save_dir=path,
                engine_path=engine_path,
                depth=depth,
                is_verbose=is_verbose,
                data_dir=path
            )
            
            if not df_date.empty:
                all_dataframes.append(df_date)
            
            # Move to next date
            current_date += timedelta(days=1)
    else:
        # Load existing analysis CSV files from data/games_analysis folder
        # Files are named with pattern: yyyy-mm-dd_game_id.csv
        path_obj = Path(path)
        
        # Look for CSV files in games_analysis folder
        games_analysis_dir = path_obj / 'games_analysis'
        
        if not games_analysis_dir.exists():
            if is_verbose:
                print(f"Directory {games_analysis_dir} does not exist.")
                print(f"Current working directory: {Path.cwd()}")
            return pd.DataFrame()
        
        if is_verbose:
            print(f"Loading CSV files from: {games_analysis_dir}")
            print(f"Date range: {start_date_str} to {end_date_str}")
        
        # Find all CSV files in the date range
        all_csv_files = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            # Pattern: yyyy-mm-dd_game_*.csv
            pattern = f"{date_str}_game_*.csv"
            csv_files = list(games_analysis_dir.glob(pattern))
            all_csv_files.extend(csv_files)
            current_date += timedelta(days=1)
        
        if not all_csv_files:
            if is_verbose:
                print(f"No CSV files found for date range {start_date_str} to {end_date_str}.")
            return pd.DataFrame()
        
        if is_verbose:
            print(f"\n{'=' * 80}")
            print(f"Found {len(all_csv_files)} CSV files to load")
            print(f"{'=' * 80}")
        
        # Load each CSV file and concatenate
        all_moves_data = []
        for csv_file in sorted(all_csv_files):
            if is_verbose:
                print(f"Loading {csv_file.name}...")
            
            try:
                df_moves = pd.read_csv(csv_file, encoding='utf-8')
                
                # Convert datetime columns if they exist
                for col in df_moves.columns:
                    if 'time' in col.lower() and df_moves[col].dtype == 'object':
                        try:
                            df_moves[col] = pd.to_datetime(df_moves[col])
                        except:
                            pass
                
                if not df_moves.empty:
                    all_moves_data.append(df_moves)
                    
                    if is_verbose:
                        print(f"  ✅ Loaded {len(df_moves)} moves")
            
            except Exception as e:
                if is_verbose:
                    print(f"  ❌ Error loading {csv_file.name}: {e}")
                continue
        
        if all_moves_data:
            df_date = pd.concat(all_moves_data, ignore_index=True)
            all_dataframes.append(df_date)
    
    # Combine all dataframes
    if not all_dataframes:
        if is_verbose:
            print("\nNo games were found or analyzed.")
        return pd.DataFrame()
    
    df_all = pd.concat(all_dataframes, ignore_index=True)
    
    if is_verbose:
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}")
        print(f"✅ Total moves analyzed: {len(df_all)}")
        print(f"📊 DataFrame shape: {df_all.shape}")
        print(f"📋 Columns: {len(df_all.columns)}")
        if 'eco_code' in df_all.columns:
            eco_counts = df_all['eco_code'].value_counts()
            print(f"📚 Unique ECO codes: {len(eco_counts)}")
    
    return df_all


def get_data(date_start, date_finish, user_name, is_verbose=True):
    """
    Get data from chess.com API and save PGN files to data/pgn folder and metadata to data/game_meta folder.
    
    Parameters:
    -----------
    date_start : str
        Start date in format 'YYYY-MM-DD' (inclusive)
    date_finish : str
        End date in format 'YYYY-MM-DD' (exclusive - not included in range)
    user_name : str
        Chess.com username
    is_verbose : bool
        If True, print detailed progress information (default: True)
    
    Returns:
    --------
    int: Number of PGN files saved
    """
    start_date = datetime.strptime(date_start, '%Y-%m-%d')
    end_date = datetime.strptime(date_finish, '%Y-%m-%d')
    
    # Use default data directory
    data_dir = DEFAULT_DATA_DIR
    
    # Create directories
    pgn_dir = data_dir / 'pgn'
    pgn_dir.mkdir(parents=True, exist_ok=True)
    game_meta_dir = data_dir / 'game_meta'
    game_meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Chess.com API requires User-Agent header to avoid 403 errors
    headers = {
        'User-Agent': 'ChessGameAnalyzer/1.0 (https://github.com/yourusername/chess-analyzer)'
    }
    
    total_saved = 0
    
    # Collect all year-month combinations in the date range
    # Note: date_finish is exclusive (not included in the range)
    year_months = set()
    temp_date = start_date
    while temp_date < end_date:
        year_month = (temp_date.year, temp_date.month)
        year_months.add(year_month)
        # Move to first day of next month
        if temp_date.month == 12:
            temp_date = temp_date.replace(year=temp_date.year + 1, month=1, day=1)
        else:
            temp_date = temp_date.replace(month=temp_date.month + 1, day=1)
        # Make sure we don't go past end_date
        if temp_date >= end_date:
            break
    
    if is_verbose:
        print(f"\nWill process {len(year_months)} month(s): {sorted(year_months)}")
    
    # Simple for-loop through year-month tuples
    for year, month in sorted(year_months):
        if is_verbose:
            print(f"\n{'=' * 80}")
            print(f"Processing {year}-{month:02d}")
            print(f"{'=' * 80}")
        
        json_url = f"https://api.chess.com/pub/player/{user_name}/games/{year}/{month:02d}"
        
        try:
            response = requests.get(json_url, headers=headers, timeout=30)
            # Check response status
            if response.status_code != 200:
                if is_verbose:
                    print(f"❌ Request failed for {year}-{month:02d}: Status {response.status_code}")
                continue
            response.raise_for_status()
            
            data = response.json()
            games_list = data.get('games', [])
            
            if not games_list:
                if is_verbose:
                    print(f"No games found for {user_name} in {year}-{month:02d}")
                continue
            
            if is_verbose:
                print(f"Found {len(games_list)} games")
            
            month_saved = 0
            # Process each game
            for i, game in enumerate(games_list):
                try:
                    # Extract unique game ID from URL
                    game_url = game.get('url', '')
                    game_id_match = re.search(r'/(\d+)$', game_url)
                    if game_id_match:
                        unique_game_id = game_id_match.group(1)
                    else:
                        # Fallback: use index if can't extract from URL
                        unique_game_id = f"{user_name}_{year}{month:02d}_{i+1:04d}"
                    
                    # Get PGN content
                    pgn_content = game.get('pgn', '')
                    
                    if not pgn_content:
                        continue
                    
                    # Extract game date from PGN headers
                    game_date_str = None
                    date_match = re.search(r'\[Date\s+"([^"]+)"\]', pgn_content)
                    if date_match:
                        pgn_date_str = date_match.group(1)
                        try:
                            # Convert from PGN format (YYYY.MM.DD) to our format (YYYY-MM-DD)
                            pgn_date = datetime.strptime(pgn_date_str, '%Y.%m.%d')
                            game_date_str = pgn_date.strftime('%Y-%m-%d')
                            
                            # Check if date is in range (date_finish is exclusive)
                            if pgn_date < start_date or pgn_date >= end_date:
                                continue
                        except:
                            continue
                    
                    if not game_date_str:
                        continue
                    
                    # Save PGN file with pattern: yyyy-mm-dd_game_id.pgn
                    filename = f"{game_date_str}_game_{unique_game_id}.pgn"
                    filepath = pgn_dir / filename
                    
                    # Skip if file already exists
                    if filepath.exists():
                        if is_verbose:
                            print(f"  [{i+1}/{len(games_list)}] Skipped (exists): {filename}")
                        continue
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(pgn_content)
                    
                    # Extract and save game metadata
                    white = game.get('white', {})
                    black = game.get('black', {})
                    
                    # Calculate points based on result
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
                    end_date_obj = datetime.fromtimestamp(end_time) if end_time else None
                    
                    # Build game metadata record
                    game_meta = {
                        'game_id': unique_game_id,
                        'game_date': game_date_str,
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
                        'end_time': end_date_obj,
                        'end_timestamp': end_time,
                        'rated': game.get('rated', False),
                    }
                    
                    # Save metadata to CSV
                    meta_filename = f"{game_date_str}_game_{unique_game_id}.csv"
                    meta_filepath = game_meta_dir / meta_filename
                    meta_df = pd.DataFrame([game_meta])
                    meta_df.to_csv(meta_filepath, index=False, encoding='utf-8')
                    
                    total_saved += 1
                    month_saved += 1
                    if is_verbose:
                        print(f"  [{i+1}/{len(games_list)}] Saved: {filename} (metadata saved)")
                
                except Exception as e:
                    if is_verbose:
                        print(f"  ⚠️  Error processing game {i+1}/{len(games_list)}: {e}")
                    continue
            
            if is_verbose and month_saved > 0:
                print(f"\n✅ Saved {month_saved} games from {year}-{month:02d}")
        
        except requests.exceptions.HTTPError as e:
            if is_verbose:
                status_code = e.response.status_code if e.response else 'Unknown'
                print(f"❌ HTTP Error for {year}-{month:02d}: Status {status_code}, Error: {e}")
            continue
        except requests.exceptions.RequestException as e:
            if is_verbose:
                print(f"❌ Request Error for {year}-{month:02d}: {e}")
            continue
        except Exception as e:
            if is_verbose:
                print(f"❌ Error processing {year}-{month:02d}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    if is_verbose:
        print(f"\n✅ Total PGN files saved: {total_saved}")
    
    return total_saved


def analyze_game(path, engine_path=None, is_verbose=True, depth=10):
    """
    Analyze a single game and save the result to data/games_analysis folder.
    Merges game metadata from data/game_meta folder if available.
    
    Parameters:
    -----------
    path : str or Path
        Path to PGN file
    engine_path : str, optional
        Path to chess engine executable. If None, will try to find Stockfish automatically.
    is_verbose : bool
        If True, print detailed progress information (default: True)
    depth : int
        Engine search depth (default: 10)
    
    Returns:
    --------
    bool: True if analysis was successful, False otherwise
    """
    pgn_path = Path(path)
    
    if not pgn_path.exists():
        if is_verbose:
            print(f"❌ PGN file not found: {pgn_path}")
        return False
    
    # Extract game date and ID from filename (pattern: yyyy-mm-dd_game_id.pgn)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', pgn_path.stem)
    game_id_match = re.search(r'game_(\d+)', pgn_path.stem)
    
    if not date_match or not game_id_match:
        if is_verbose:
            print(f"❌ Cannot extract date or game_id from filename: {pgn_path.name}")
        return False
    
    game_date_str = date_match.group(1)
    game_id = game_id_match.group(1)
    
    # Check if analysis already exists
    games_analysis_dir = DEFAULT_DATA_DIR / 'games_analysis'
    games_analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = games_analysis_dir / f"{game_date_str}_game_{game_id}.csv"
    
    if csv_file_path.exists():
        if is_verbose:
            print(f"⏭️  Analysis already exists: {csv_file_path.name}")
        return True
    
    if is_verbose:
        print(f"Analyzing {pgn_path.name}...")
    
    try:
        # Load game metadata if available
        game_meta_dir = DEFAULT_DATA_DIR / 'game_meta'
        meta_filepath = game_meta_dir / f"{game_date_str}_game_{game_id}.csv"
        
        game_meta = None
        if meta_filepath.exists():
            try:
                meta_df = pd.read_csv(meta_filepath, encoding='utf-8')
                if not meta_df.empty:
                    game_meta = meta_df.iloc[0].to_dict()
                    # Convert end_time to datetime if it's a string
                    if 'end_time' in game_meta and isinstance(game_meta['end_time'], str):
                        try:
                            game_meta['end_time'] = pd.to_datetime(game_meta['end_time'])
                        except:
                            pass
            except Exception as e:
                if is_verbose:
                    print(f"  ⚠️  Could not load metadata: {e}")
        
        # Use the existing analyze_pgn_evaluations function
        df = analyze_pgn_evaluations(
            pgn_path,
            engine_path=engine_path,
            depth=depth,
            data_dir=DEFAULT_DATA_DIR  # This will save to data/games_analysis
        )
        
        if df.empty:
            if is_verbose:
                print(f"  ⚠️  No moves found in game")
            return False
        
        # Merge metadata with analysis DataFrame
        if game_meta:
            for key, value in game_meta.items():
                df[key] = value
        
        # Save the merged DataFrame
        games_analysis_dir = DEFAULT_DATA_DIR / 'games_analysis'
        games_analysis_dir.mkdir(parents=True, exist_ok=True)
        csv_file_path = games_analysis_dir / f"{game_date_str}_game_{game_id}.csv"
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        
        if is_verbose:
            print(f"  ✅ Analyzed {len(df)} moves")
            if game_meta:
                print(f"  ✅ Merged with metadata")
        
        return True
    
    except Exception as e:
        if is_verbose:
            print(f"  ❌ Error analyzing game: {e}")
            import traceback
            traceback.print_exc()
        return False


def analyze_games(date_start, date_finish, engine_path=None, is_verbose=True, depth=10):
    """
    Launch analyze_game() for all games in the date range.
    
    Parameters:
    -----------
    date_start : str
        Start date in format 'YYYY-MM-DD' (inclusive)
    date_finish : str
        End date in format 'YYYY-MM-DD' (exclusive - not included in range)
    user_name : str
        Chess.com username (used to filter PGN files if needed)
    engine_path : str, optional
        Path to chess engine executable. If None, will try to find Stockfish automatically.
    is_verbose : bool
        If True, print detailed progress information (default: True)
    depth : int
        Engine search depth (default: 10)
    
    Returns:
    --------
    dict: Statistics with 'total', 'analyzed', 'skipped', 'errors'
    """
    start_date = datetime.strptime(date_start, '%Y-%m-%d')
    end_date = datetime.strptime(date_finish, '%Y-%m-%d')
    
    # Find all PGN files in the date range
    pgn_dir = DEFAULT_DATA_DIR / 'pgn'
    
    if not pgn_dir.exists():
        if is_verbose:
            print(f"❌ PGN directory does not exist: {pgn_dir}")
        return {'total': 0, 'analyzed': 0, 'skipped': 0, 'errors': 0}
    
    # Find all PGN files in the date range (date_finish is exclusive)
    all_pgn_files = []
    current_date = start_date
    while current_date < end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        pattern = f"{date_str}_game_*.pgn"
        pgn_files = list(pgn_dir.glob(pattern))
        all_pgn_files.extend(pgn_files)
        current_date += timedelta(days=1)
    
    if not all_pgn_files:
        if is_verbose:
            print(f"No PGN files found for date range {date_start} to {date_finish}")
        return {'total': 0, 'analyzed': 0, 'skipped': 0, 'errors': 0}
    
    if is_verbose:
        print(f"\n{'=' * 80}")
        print(f"Found {len(all_pgn_files)} PGN files to analyze")
        print(f"{'=' * 80}")
    
    stats = {'total': len(all_pgn_files), 'analyzed': 0, 'skipped': 0, 'errors': 0}
    
    # Analyze each game
    for i, pgn_file in enumerate(sorted(all_pgn_files), 1):
        if is_verbose:
            print(f"\n[{i}/{len(all_pgn_files)}] ", end='')
        
        # Check if analysis already exists before analyzing
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', pgn_file.stem)
        game_id_match = re.search(r'game_(\d+)', pgn_file.stem)
        
        if date_match and game_id_match:
            csv_file = DEFAULT_DATA_DIR / 'games_analysis' / f"{date_match.group(1)}_game_{game_id_match.group(1)}.csv"
            if csv_file.exists():
                stats['skipped'] += 1
                if is_verbose:
                    print(f"⏭️  Skipped (already exists): {pgn_file.name}")
                continue
        
        # Analyze the game
        result = analyze_game(pgn_file, engine_path=engine_path, is_verbose=is_verbose, depth=depth)
        
        if result:
            stats['analyzed'] += 1
        else:
            stats['errors'] += 1
    
    if is_verbose:
        print(f"\n{'=' * 80}")
        print("ANALYSIS SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total games: {stats['total']}")
        print(f"Analyzed: {stats['analyzed']}")
        print(f"Skipped (already exists): {stats['skipped']}")
        print(f"Errors: {stats['errors']}")
    
    return stats


def get_analysys_results(date_start, date_finish, is_verbose=True, api=0, user_name=None, engine_path=None, depth=10):
    """
    Concatenate CSV files from data/games_analysis folder by date range.
    If api=1, first downloads all games from chess.com API and analyzes them.
    
    Parameters:
    -----------
    date_start : str
        Start date in format 'YYYY-MM-DD' (inclusive)
    date_finish : str
        End date in format 'YYYY-MM-DD' (exclusive - not included in range)
    is_verbose : bool
        If True, print detailed progress information (default: True)
    api : int
        If 1, download all games from chess.com API and analyze them first (default: 0)
    user_name : str, optional
        Chess.com username (required if api=1)
    engine_path : str, optional
        Path to chess engine executable (required if api=1). If None, will try to find Stockfish automatically.
    depth : int
        Engine search depth (default: 10, used if api=1)
    
    Returns:
    --------
    pd.DataFrame: Concatenated DataFrame with all moves from all games in the date range
    """
    # If api=1, download all batches and analyze them first
    if api == 1:
        if user_name is None:
            raise ValueError("user_name is required when api=1")
        
        if is_verbose:
            print("=" * 80)
            print("STEP 1: DOWNLOADING GAMES FROM CHESS.COM API")
            print("=" * 80)
        
        # Download all games in the date range
        get_data(date_start, date_finish, user_name, is_verbose=is_verbose)
        
        if is_verbose:
            print("\n" + "=" * 80)
            print("STEP 2: ANALYZING DOWNLOADED GAMES")
            print("=" * 80)
        
        # Analyze all downloaded games
        analyze_games(date_start, date_finish, engine_path=engine_path, is_verbose=is_verbose, depth=depth)
        
        if is_verbose:
            print("\n" + "=" * 80)
            print("STEP 3: LOADING ANALYSIS RESULTS")
            print("=" * 80)
    start_date = datetime.strptime(date_start, '%Y-%m-%d')
    end_date = datetime.strptime(date_finish, '%Y-%m-%d')
    
    # Find all CSV files in the date range
    games_analysis_dir = DEFAULT_DATA_DIR / 'games_analysis'
    
    if not games_analysis_dir.exists():
        if is_verbose:
            print(f"❌ Analysis directory does not exist: {games_analysis_dir}")
        return pd.DataFrame()
    
    if is_verbose:
        print(f"Loading CSV files from: {games_analysis_dir}")
        print(f"Date range: {date_start} to {date_finish}")
    
    # Find all CSV files in the date range (date_finish is exclusive)
    all_csv_files = []
    current_date = start_date
    while current_date < end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        pattern = f"{date_str}_game_*.csv"
        csv_files = list(games_analysis_dir.glob(pattern))
        all_csv_files.extend(csv_files)
        current_date += timedelta(days=1)
    
    if not all_csv_files:
        if is_verbose:
            print(f"No CSV files found for date range {date_start} to {date_finish}")
        return pd.DataFrame()
    
    if is_verbose:
        print(f"\n{'=' * 80}")
        print(f"Found {len(all_csv_files)} CSV files to load")
        print(f"{'=' * 80}")
    
    # Load metadata directory for merging
    game_meta_dir = DEFAULT_DATA_DIR / 'game_meta'
    
    # Load each CSV file and concatenate
    all_dataframes = []
    for i, csv_file in enumerate(sorted(all_csv_files), 1):
        if is_verbose:
            print(f"[{i}/{len(all_csv_files)}] Loading {csv_file.name}...")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Convert datetime columns if they exist
            for col in df.columns:
                if 'time' in col.lower() and df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='raise')
                    except:
                        pass
            
            # Extract game date and ID from filename to load metadata
            csv_stem = csv_file.stem  # e.g., "2025-07-15_game_140721976858"
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', csv_stem)
            game_id_match = re.search(r'game_(\d+)', csv_stem)
            
            # Merge metadata if available and not already in dataframe
            if date_match and game_id_match:
                game_date_str = date_match.group(1)
                game_id = game_id_match.group(1)
                meta_filepath = game_meta_dir / f"{game_date_str}_game_{game_id}.csv"
                
                if meta_filepath.exists():
                    try:
                        meta_df = pd.read_csv(meta_filepath, encoding='utf-8')
                        if not meta_df.empty:
                            game_meta = meta_df.iloc[0].to_dict()
                            
                            # Only add metadata columns that are missing or have null values
                            for key, value in game_meta.items():
                                if key not in df.columns or df[key].isna().all():
                                    df[key] = value
                            
                            # Convert end_time to datetime if it's a string
                            if 'end_time' in df.columns and df['end_time'].dtype == 'object':
                                try:
                                    df['end_time'] = pd.to_datetime(df['end_time'])
                                except:
                                    pass
                    except Exception as e:
                        if is_verbose:
                            print(f"  ⚠️  Could not load metadata for {csv_file.name}: {e}")
            
            if not df.empty:
                all_dataframes.append(df)
                
                if is_verbose:
                    print(f"  ✅ Loaded {len(df)} moves")
        
        except Exception as e:
            if is_verbose:
                print(f"  ❌ Error loading {csv_file.name}: {e}")
            continue
    
    if not all_dataframes:
        if is_verbose:
            print("\nNo data was loaded.")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    df_all = pd.concat(all_dataframes, ignore_index=True)
    
    if is_verbose:
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}")
        print(f"✅ Total moves loaded: {len(df_all)}")
        print(f"📊 DataFrame shape: {df_all.shape}")
        print(f"📋 Columns: {len(df_all.columns)}")
        if 'eco_code' in df_all.columns:
            eco_counts = df_all['eco_code'].value_counts()
            print(f"📚 Unique ECO codes: {len(eco_counts)}")
    
    return df_all

def make_user_df(df):
    """
    Make a user base dataframe from the analysis results.
    Parameters:
    -----------
    df : pd.DataFrame
        Analysis results dataframe
    
    Returns:
    --------
    pd.DataFrame: User base dataframe
    """

    white_df = df[df['player'] == 'White'].copy()
    white_df['color'] ='White'
    white_df['player'] = white_df['white_username']
    white_df['opponent'] = white_df['black_username']

    white_df['result'] = white_df['white_result']
    white_df['opponent_result'] = white_df['black_result']


    white_df['points'] = white_df['white_points']
    white_df['opponent_points'] = white_df['black_points']
    
    white_df['rating'] = white_df['white_rating']
    white_df['opponent_rating'] = white_df['black_rating']

    white_df['time_remaining'] = white_df['white_time_remaining']
    white_df['opponent_time_remaining'] = white_df['black_time_remaining']

    white_df['time_used'] = white_df['white_time_used']
    white_df['opponent_time_used'] = white_df['black_time_used']

    white_df['material'] = white_df['white_material']
    white_df['opponent_material'] = white_df['black_material']

    white_df['pawns'] = white_df['white_pawns']
    white_df['opponent_pawns'] = white_df['black_pawns']

    white_df['isolated_pawns'] = white_df['white_isolated_pawns']
    white_df['opponent_isolated_pawns'] = white_df['black_isolated_pawns']

    white_df['doubled_pawns'] = white_df['white_doubled_pawns']
    white_df['opponent_doubled_pawns'] = white_df['black_doubled_pawns']

    white_df['passed_pawns'] = white_df['white_passed_pawns']
    white_df['opponent_passed_pawns'] = white_df['black_passed_pawns']

    white_df['center_control'] = white_df['white_center_control']
    white_df['opponent_center_control'] = white_df['black_center_control']

    white_df['castled'] = white_df['white_castled']
    white_df['opponent_castled'] = white_df['white_castled']

    white_df['captures'] = white_df['white_captures']
    white_df['opponent_captures'] = white_df['black_captures']

    white_df['promotions'] = white_df['white_promotions']
    white_df['opponent_promotions'] = white_df['black_promotions']

    black_df = df[df['player'] == 'Black'].copy()
    black_df['color'] = 'Black'
    black_df['player'] = black_df['black_username']
    black_df['opponent'] = black_df['white_username']

    black_df['result'] = black_df['black_result']
    black_df['opponent_result'] = black_df['white_result']

    black_df['points'] = black_df['black_points']
    black_df['opponent_points'] = black_df['white_points']

    black_df['rating'] = black_df['black_rating']
    black_df['opponent_rating'] = black_df['white_rating']

    black_df['time_remaining'] = black_df['black_time_remaining']
    black_df['opponent_time_remaining'] = black_df['white_time_remaining']

    black_df['time_used'] = black_df['black_time_used']
    black_df['opponent_time_used'] = black_df['white_time_used']

    black_df['material'] = black_df['black_material']
    black_df['opponent_material'] = black_df['white_material']

    black_df['pawns'] = black_df['black_pawns']
    black_df['opponent_pawns'] = black_df['white_pawns']

    black_df['isolated_pawns'] = black_df['black_isolated_pawns']
    black_df['opponent_isolated_pawns'] = black_df['white_isolated_pawns']

    black_df['doubled_pawns'] = black_df['black_doubled_pawns']
    black_df['opponent_doubled_pawns'] = black_df['white_doubled_pawns']

    black_df['passed_pawns'] = black_df['black_passed_pawns']
    black_df['opponent_passed_pawns'] = black_df['white_passed_pawns']

    black_df['center_control'] = black_df['black_center_control']
    black_df['opponent_center_control'] = black_df['white_center_control']

    black_df['castled'] = black_df['black_castled']
    black_df['opponent_castled'] = black_df['white_castled']

    black_df['captures'] = black_df['black_captures']
    black_df['opponent_captures'] = black_df['white_captures']

    black_df['promotions'] = black_df['black_promotions']
    black_df['opponent_promotions'] = black_df['white_promotions']

    user_base_df = pd.concat([white_df, black_df])

    user_base_df['game_end_time'] = user_base_df['end_time']
    user_base_df['rating_difference'] = user_base_df['rating'] - user_base_df['opponent_rating']
    user_base_df['material_balance'] = user_base_df['material'] - user_base_df['opponent_material']

    user_base_df = user_base_df[[
        'game_id',
        'game_end_time',
        'color',
        'player',
        'opponent',
        'result',
        'opponent_result',
        'points',
        'opponent_points',
        'rating',
        'opponent_rating',
        'rating_difference',
        'time_control', 'time_class', 'rules', 'rated',
        'url',
        'game_phase',
        'move_number', 'full_move', 'move_san',
        'evaluation', 'evaluation_pawns', 'eval_change',
        'move_type',
        'time_remaining',
        'opponent_time_remaining',
        'time_used',
        'opponent_time_used',
        'is_same_material_kind', 
        'material', 'opponent_material', 'material_balance',
        'is_capture', 'is_promotion',
        'pawns',
        'opponent_pawns', 'isolated_pawns', 'opponent_isolated_pawns',
        'doubled_pawns', 'opponent_doubled_pawns', 'passed_pawns',
        'opponent_passed_pawns', 'center_control', 'opponent_center_control',
        'castled', 'opponent_castled', 'captures', 'opponent_captures',
        'promotions', 'opponent_promotions'

    ]]

    user_base_df['game_id'] = user_base_df['game_id'].astype(str)
    user_base_df['move_number'] = user_base_df['move_number'].astype(str)
    user_base_df['full_move'] = user_base_df['full_move'].astype(str)

    user_base_df['evaluation_pawns_relative'] = np.where(
                                                            user_base_df['color'] == 'White', 
                                                            user_base_df['evaluation_pawns'],
                                                            -user_base_df['evaluation_pawns']
                                                        )
    user_base_df['evaluation_pawns_group'] = np.round(user_base_df['evaluation_pawns_relative'], 0).astype(int)
    user_base_df['evaluation_pawns_relative_group'] = np.round(user_base_df['evaluation_pawns_relative'], 0).astype(int)
    user_base_df['date'] = user_base_df['game_end_time'].dt.date
    user_base_df['week'] = (user_base_df['game_end_time'] - pd.to_timedelta(user_base_df['game_end_time'].dt.weekday, unit='d')).dt.date
    
    return user_base_df
    


def get_move_stat(work_df, fields=['color']):
    """
    Get statistics from the work dataframe.
    One row = one players move in one game.
    Parameters:
    -----------
    work_df : pd.DataFrame
        Work dataframe
    fields : list
        Fields to group by. 
        Player adds in fields automatically.
    Returns:
    --------
    pd.DataFrame: Statistics dataframe
    """
    full_fields = fields.copy()
    full_fields.append('player')
    full_fields.reverse()

    stat_df = work_df.groupby(full_fields).agg(
        games = ('game_id', 'nunique'),
        moves_total = ('move_number', 'count'),
        points = ('points', 'mean'),
        rating = ('rating', 'mean'),
        opponent_rating = ('opponent_rating', 'mean'),
        rating_difference = ('rating_difference', 'mean'),
        time_used_total = ('time_used', 'max'),
        time_remaining_avg = ('time_remaining', 'mean'),
        material_balance_max = ('material_balance', 'max'),
        material_balance_min = ('material_balance', 'min'),
        normal_moves = ('move_type', lambda x: (x == 'normal').sum()),
        inaccuracy_moves = ('move_type', lambda x: (x == 'inaccuracy').sum()),
        blunder_moves = ('move_type', lambda x: (x == 'blunder').sum()),
        mistake_moves = ('move_type', lambda x: (x == 'mistake').sum()),
        good_moves = ('move_type', lambda x: (x == 'good').sum()),
        eval_avg = ('evaluation_pawns_relative', 'mean'),
        last_eval = ('evaluation_pawns_relative', 'last'),
        eval_change_total = ('eval_change', 'sum'),
    ).reset_index()
    stat_df['accur'] = stat_df['eval_change_total'] / stat_df['moves_total']
    # magic constant
    # kx+b in this logic
    # accur = 3 (ist line) = 1
    # accur = -7 (lose material every move) = 0
    stat_df['accur'] = 10/13/10 * stat_df['accur'] + 10/13
    stat_df['accur'] = np.where(
        stat_df['accur'] < 0, 
        0, 
        stat_df['accur']
        )
    stat_df['accur'] = np.where(
        stat_df['accur'] > 1, 
        1, 
        stat_df['accur']
        )
    

    return stat_df


def get_player_stat(df, fields=['color']):
    """
    Get statistics from the dataframe.
    One row = one player.
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    fields : list 
        Fields to group by. 
        Player adds in fields automatically.
    Returns:
    --------
    pd.DataFrame: Statistics dataframe
    """
    player_fileds = fields.copy()
    player_fileds.append('game_id')
    player_fileds.append('opponent')

    right_fileds = fields.copy()
    right_fileds.append('opponent')
    right_fileds.append('game_id')

    left_fileds = fields.copy()
    left_fileds.append('player')
    left_fileds.append('game_id')

    # quick fix for color
    for col in right_fileds:
        if col == 'color':
            right_fileds.remove(col)
    for col in left_fileds:
        if col == 'color':
            left_fileds.remove(col)

    opponent_fileds = right_fileds.copy()
    opponent_fileds.append('accur')

    game_stat_df_base = get_move_stat(df, fields=player_fileds)
    game_stat_df = game_stat_df_base.merge(
        game_stat_df_base[opponent_fileds],
        'left',
        left_on=left_fileds,
        right_on=right_fileds,
        suffixes=('', '_opponent')
    )
    del game_stat_df['opponent_opponent']

    game_stat_df['xG'] = game_stat_df['accur'] - game_stat_df['accur_opponent']

    player_stat_fields = fields.copy()
    player_stat_fields.append('player')
    player_stat_fields.reverse()

    player_stat_df = game_stat_df.groupby(player_stat_fields).agg(
        games = ('game_id', 'nunique'),
        moves = ('moves_total', 'sum'),
        winrate = ('points', 'mean'),
        rating = ('rating', 'mean'),
        opponent_rating = ('opponent_rating', 'mean'),
        accur = ('accur', 'mean'),
        accur_opponent = ('accur_opponent', 'mean'),
        xG = ('xG', 'mean'),
        time_used_avg = ('time_used_total', 'mean'),
        time_remaining_avg = ('time_remaining_avg', 'mean'),
        good_moves = ('good_moves', 'sum'),
        normal_moves = ('normal_moves', 'sum'),
        inaccuracy_moves = ('inaccuracy_moves', 'sum'),
        mistake_moves = ('mistake_moves', 'sum'),
        blunder_moves = ('blunder_moves', 'sum'),
        last_eval = ('last_eval', 'mean'),
    ).reset_index()
    player_stat_df['rating_difference'] = player_stat_df['rating'] - player_stat_df['opponent_rating']
    player_stat_df['moves_per_game'] = player_stat_df['moves'] / player_stat_df['games']

    player_stat_df['good_moves'] = player_stat_df['good_moves'] / player_stat_df['moves']
    player_stat_df['normal_moves'] = player_stat_df['normal_moves'] / player_stat_df['moves']
    player_stat_df['inaccuracy_moves'] = player_stat_df['inaccuracy_moves'] / player_stat_df['moves']
    player_stat_df['mistake_moves'] = player_stat_df['mistake_moves'] / player_stat_df['moves']
    player_stat_df['blunder_moves'] = player_stat_df['blunder_moves'] / player_stat_df['moves']

    return player_stat_df



def get_adv_cap(work_df, fields=[], brackets=(-4, 4)):
    """
    Get advanced capitallisation statistics from the work dataframe.
    Parameters:
    -----------
    work_df : pd.DataFrame
        Work dataframe
    fields : list
        Fields to group by. 
        Player adds in fields automatically.
    brackets : tuple
        Brackets for evaluation pawns relative group.
    Returns:
    --------
    pd.DataFrame: Advanced wide capitallisation dataframe
    pd.DataFrame: Advanced capitallisation statistics in owe row
    pd.DataFrame: Resoursefillness statistics in own row
    """
    full_fields = fields.copy()
    full_fields.append('evaluation_pawns_relative_group')
    adv_cap_df = get_player_stat(work_df, fields=full_fields)
    adv_cap_df = adv_cap_df[
                    (adv_cap_df['evaluation_pawns_relative_group'] >= brackets[0]) &
                    (adv_cap_df['evaluation_pawns_relative_group'] <= brackets[1])
                    ]
    
    base_fields = fields.copy()
    base_fields.append('player')
    base_fields.append('games')
    adv_cap_df_base = get_player_stat(work_df, fields=fields)[base_fields]
    join_fields = fields.copy()
    join_fields.append('player')
    adv_cap_df = adv_cap_df.merge(adv_cap_df_base, 'left', on=join_fields, suffixes=('', '_base'))
    adv_cap_df['share'] = adv_cap_df['games'] / adv_cap_df['games_base']

    adv_cap_stat = adv_cap_df[adv_cap_df['evaluation_pawns_relative_group'] > 0].groupby(join_fields).agg(
        share = ('share', 'mean'),
        winrate = ('winrate', 'mean'),
    ).reset_index()

    res_stat = adv_cap_df[adv_cap_df['evaluation_pawns_relative_group'] < 0].groupby(join_fields).agg(
        share = ('share', 'mean'),
        winrate = ('winrate', 'mean'),
    ).reset_index()
    return adv_cap_df, adv_cap_stat, res_stat