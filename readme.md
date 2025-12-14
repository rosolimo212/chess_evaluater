# Chess Evaluater

A comprehensive Python toolkit for downloading chess games from Chess.com, analyzing them with Stockfish engine, and generating detailed statistics and insights.

## Overview

This project provides tools to:
- Download games from Chess.com API
- Analyze games using Stockfish chess engine
- Generate move-by-move evaluations and statistics
- Create player performance metrics and insights

## Features

- **Game Download**: Fetch games from Chess.com API for any user and date range
- **Engine Analysis**: Analyze games with Stockfish at configurable depth
- **Move Evaluation**: Get detailed move-by-move evaluations, including:
  - Position evaluations
  - Move classifications (good, normal, inaccuracy, mistake, blunder)
  - Material balance
  - Pawn structure analysis
  - Time usage statistics
- **Statistics**: Generate player statistics including winrate, accuracy, and performance metrics
- **Data Persistence**: All data is saved in a structured format for easy access

## Installation

### Requirements

- Python 3.7+
- Stockfish chess engine (will be auto-detected if in PATH)
- Required Python packages:
  ```bash
  pip install pandas numpy requests python-chess
  ```

### Installing Stockfish

**Linux:**
```bash
sudo apt-get install stockfish  # Debian/Ubuntu
sudo yum install stockfish       # CentOS/RHEL
```

**macOS:**
```bash
brew install stockfish
```

**Windows:**
Download from [Stockfish official website](https://stockfishchess.org/download/) and add to PATH

## Data Structure

All data is stored in `/home/roman/python/kotelok/chess_evaluater/data/` with the following structure:

```
data/
├── pgn/                    # PGN game files
│   └── YYYY-MM-DD_game_ID.pgn
├── game_meta/              # Game metadata CSV files
│   └── YYYY-MM-DD_game_ID.csv
└── games_analysis/         # Move-by-move analysis CSV files
    └── YYYY-MM-DD_game_ID.csv
```

## Main Functions

### Core Functions

#### `get_data(date_start, date_finish, user_name, is_verbose=True)`
Downloads games from Chess.com API for a given date range and user.
- Saves PGN files to `data/pgn/`
- Saves metadata to `data/game_meta/`

#### `analyze_games(date_start, date_finish, engine_path=None, is_verbose=True, depth=10)`
Analyzes all PGN files in the date range using Stockfish.
- Reads PGN files from `data/pgn/`
- Saves analysis to `data/games_analysis/`
- Returns statistics dictionary

#### `get_analysys_results(date_start, date_finish, is_verbose=True, api=0, user_name=None, engine_path=None, depth=10)`
Main function to get analysis results. Can optionally download and analyze games first.
- **api=0** (default): Only loads existing CSV files from `data/games_analysis/`
- **api=1**: Downloads games from API, analyzes them, then loads results
- Returns concatenated DataFrame with all moves from all games

#### `make_user_df(df)`
Transforms analysis DataFrame into user-focused format with player/opponent perspectives.

#### `get_player_stat(df, fields=['color'])`
Generates player statistics grouped by specified fields (e.g., color, time_class).

#### `get_move_stat(work_df, fields=['color'])`
Generates move-level statistics grouped by specified fields.

### Utility Functions

#### `fetch_chess_com_games(username, date, save_dir=None, is_verbose=True)`
Fetches games for a specific user and date (single month).

#### `analyze_pgn_evaluations(pgn, engine_path=None, depth=10, data_dir=None)`
Analyzes a single PGN game file.

## Usage Examples

### Basic Usage - Load Existing Data

```python
import chess_analyzer as chan

# Load existing analysis files
anl_res_df = chan.get_analysys_results(
    date_start='2024-12-14',
    date_finish='2025-12-14',
    is_verbose=False
)

# Transform to user-focused format
work_df = chan.make_user_df(anl_res_df)

# Get player statistics
player_stats = chan.get_player_stat(work_df, fields=['color'])
```

### Download and Analyze New Data

```python
import chess_analyzer as chan

# Download, analyze, and load results in one call
anl_res_df = chan.get_analysys_results(
    date_start='2024-12-14',
    date_finish='2025-12-14',
    is_verbose=True,
    api=1,                          # Enable API download
    user_name='YourUsername',       # Required when api=1
    engine_path=None,               # Auto-detect Stockfish
    depth=10                        # Analysis depth
)

work_df = chan.make_user_df(anl_res_df)
```

### Step-by-Step Workflow

```python
import chess_analyzer as chan

# Step 1: Download games
chan.get_data(
    date_start='2024-12-14',
    date_finish='2025-12-14',
    user_name='YourUsername',
    is_verbose=True
)

# Step 2: Analyze downloaded games
stats = chan.analyze_games(
    date_start='2024-12-14',
    date_finish='2025-12-14',
    engine_path='stockfish',  # or None for auto-detect
    depth=10,
    is_verbose=True
)

# Step 3: Load analysis results
anl_res_df = chan.get_analysys_results(
    date_start='2024-12-14',
    date_finish='2025-12-14',
    is_verbose=True
)
```

### Generate Statistics

```python
import chess_analyzer as chan
import datetime

# Load data
anl_res_df = chan.get_analysys_results('2024-12-14', '2025-12-14')
work_df = chan.make_user_df(anl_res_df)

# Filter by time period
today = datetime.datetime.now()
month_ago = today - datetime.timedelta(days=30)

month_df = work_df[
    (work_df['game_end_time'] >= month_ago) &
    (work_df['game_end_time'] <= today)
]

# Get player statistics
month_stats = chan.get_player_stat(month_df, fields=['color'])

# Get move statistics
move_stats = chan.get_move_stat(month_df, fields=['color'])
```

## File Structure

```
chess_evaluater/
├── chess_analyzer.py      # Main code file with all functions
├── tests.ipynb            # Comprehensive test suite
├── show_stat.ipynb        # Main notebook for analysis and visualization
├── readme.md              # This file
└── data/                  # Data directory (created automatically)
    ├── pgn/               # PGN game files
    ├── game_meta/         # Game metadata
    └── games_analysis/    # Analysis results
```

## Configuration

### Data Directory

The default data directory is set to:
```
/home/roman/python/kotelok/chess_evaluater/data
```

This is defined in `chess_analyzer.py` as `DEFAULT_DATA_DIR`. All functions use this path by default.

### Stockfish Configuration

The engine will auto-detect Stockfish if it's in your PATH. Otherwise, you can specify the path:

```python
chan.analyze_games(
    date_start='2024-12-14',
    date_finish='2025-12-14',
    engine_path='/path/to/stockfish',
    depth=15  # Higher depth = more accurate but slower
)
```

## Date Range Format

All date parameters use the format `'YYYY-MM-DD'`:
- `date_start`: Inclusive (included in range)
- `date_finish`: Exclusive (not included in range)

Example: `date_start='2024-12-14'` to `date_finish='2025-12-14'` includes all games from Dec 14, 2024 to Dec 13, 2025.

## Output Data Format

### Analysis DataFrame Columns

The analysis DataFrame includes:
- Game metadata: `game_id`, `game_end_time`, `player`, `opponent`, `result`, `rating`, etc.
- Move data: `move_number`, `move_san`, `evaluation`, `evaluation_pawns`, `move_type`
- Position features: `material`, `pawns`, `isolated_pawns`, `center_control`, etc.
- Time data: `time_remaining`, `time_used`, `time_control`
- Derived metrics: `evaluation_pawns_relative`, `date`, `week`

### Statistics Output

`get_player_stat()` returns statistics including:
- `games`, `moves`, `winrate`
- `rating`, `opponent_rating`, `rating_difference`
- `accur`, `accur_opponent`, `xG`
- Move type distributions: `good_moves`, `normal_moves`, `inaccuracy_moves`, etc.

## Testing

Run the test suite in `tests.ipynb` to verify:
- Folder structure integrity
- File naming conventions
- Function correctness
- API interactions
- Data consistency

## Notes

- Analysis can be time-consuming for large date ranges. Consider using `is_verbose=True` to monitor progress.
- Games are cached after analysis. Re-running analysis will skip already-analyzed games.
- The `api=1` parameter in `get_analysys_results()` will download and analyze all games in the date range, which may take significant time.

## License

See LICENSE file for details.
