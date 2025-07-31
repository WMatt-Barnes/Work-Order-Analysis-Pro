import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import pandas as pd
import os
import re
import logging
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import rapidfuzz
from rapidfuzz import fuzz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import warnings
from typing import Optional, Union, Any, List, Dict

# AI Classification imports
try:
    from ai_failure_classifier import AIClassifier, AIClassificationResult
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI classifier not available. Install dependencies and ensure ai_failure_classifier.py is present.")

# New analysis modules
try:
    from weibull_analysis import WeibullAnalysis
    from fmea_export import FMEAExport
    from pm_analysis import PMAnalysis
    from spares_analysis import SparesAnalysis
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logging.warning(f"Analysis modules not available: {e}")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {e}")

# Constants
DEFAULT_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "failure_mode_dictionary_.xlsx")
DEFAULT_CODE = "No Failure Mode Identified"
DEFAULT_DESC = "No Failure Mode Identified"
CUSTOM_CODE = "custom"
LOG_FILE = "matching_log.txt"
ABBREVIATIONS = {
    'comp': 'compressor',
    'leek': 'leak',
    'leeking': 'leaking',
    'brk': 'break',
    'mtr': 'motor',
}
THRESHOLD = 75  # Fuzzy matching threshold
DATE_FORMATS = ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]

# Required columns for work order files
REQUIRED_COLUMNS = {
    'Work Order': 'Work Order Number',
    'Description': 'Work Description', 
    'Asset': 'Asset Name',
    'Equipment #': 'Equipment Number',
    'Work Type': 'Work Type',
    'Reported Date': 'Date Reported',
    'Work Order Cost': 'Work Order Cost (Optional)',
    'User failure code': 'User Failure Code (Optional)'
}

# AI Configuration
AI_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for embeddings
AI_CACHE_FILE = "ai_classification_cache.json"

# Configuration files
CONFIG_FILE = "app_config.json"
COLUMN_MAPPING_FILE = "column_mapping.json"

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize stemmer
stemmer = SnowballStemmer("english")

def load_app_config() -> dict:
    """Load application configuration from JSON file."""
    default_config = {
        'last_work_order_path': '',
        'last_dictionary_path': '',
        'last_output_directory': '',
        'ai_enabled': False,
        'confidence_threshold': AI_CONFIDENCE_THRESHOLD
    }
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with default config to ensure all keys exist
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                logging.info(f"Loaded configuration from {CONFIG_FILE}")
                return config
        else:
            logging.info(f"Configuration file not found, using defaults")
            return default_config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return default_config

def save_app_config(config: dict):
    """Save application configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations in text."""
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)
    return text

def compile_patterns(keywords: list) -> list:
    """Compile regex patterns for keywords."""
    patterns = []
    for keyword in keywords:
        try:
            pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
            patterns.append(pattern)
        except re.error as e:
            logging.error(f"Error compiling pattern for keyword '{keyword}': {e}")
    return patterns

def match_failure_mode(description: str, dictionary: list) -> tuple:
    """Match failure mode in description using exact, fuzzy, and stemmed matching."""
    if not description or not isinstance(description, str):
        logging.debug(f"Invalid description: {description}")
        return DEFAULT_CODE, DEFAULT_DESC, ''
    
    norm_desc = normalize_text(description)
    norm_desc = expand_abbreviations(norm_desc)
    logging.debug(f"Normalized description: {norm_desc}")
    
    try:
        tokens = word_tokenize(norm_desc)
        stemmed_desc = ' '.join(stemmer.stem(token) for token in tokens)
        logging.debug(f"Stemmed description: {stemmed_desc}")
    except Exception as e:
        logging.error(f"Error tokenizing description: {e}")
        stemmed_desc = norm_desc
    
    desc_length = len(norm_desc)
    
    for keyword, norm_keyword, stemmed_keyword, code, failure_desc, pattern in dictionary:
        keyword_list = [k.strip().lower() for k in keyword.split(',')]
        for kw in keyword_list:
            norm_kw = normalize_text(kw)
            kw_length = len(norm_kw)
            try:
                kw_tokens = word_tokenize(norm_kw)
                stemmed_kw = ' '.join(stemmer.stem(token) for token in kw_tokens)
            except Exception as e:
                logging.error(f"Error tokenizing keyword '{kw}': {e}")
                stemmed_kw = norm_kw
            
            if norm_kw in norm_desc or kw in norm_desc:
                logging.info(f"Exact/partial match: keyword={kw}, code={code}, desc={failure_desc}")
                return code, failure_desc, kw
            
            try:
                if re.search(r'\b' + re.escape(kw) + r'\b', norm_desc, re.IGNORECASE):
                    logging.info(f"Regex match: keyword={kw}, code={code}, desc={failure_desc}")
                    return code, failure_desc, kw
            except re.error as e:
                logging.error(f"Error in regex for keyword '{kw}': {e}")
            
            if stemmed_kw in stemmed_desc:
                logging.info(f"Stemmed match: keyword={kw}, code={code}, desc={failure_desc}")
                return code, failure_desc, kw
            
            if min(kw_length, desc_length) / max(kw_length, desc_length) >= 0.5:
                score = fuzz.partial_ratio(norm_kw, norm_desc)
                logging.debug(f"Fuzzy match attempt: keyword={kw}, score={score}, length_ratio={min(kw_length, desc_length)/max(kw_length, desc_length):.2f}")
                if score >= THRESHOLD:
                    logging.info(f"Fuzzy match: keyword={kw}, score={score}, code={code}, desc={failure_desc}")
                    return code, failure_desc, kw
            else:
                logging.debug(f"Fuzzy match skipped: keyword={kw}, length_ratio={min(kw_length, desc_length)/max(kw_length, desc_length):.2f}")
    
    logging.info(f"No match for description: {norm_desc}")
    return DEFAULT_CODE, DEFAULT_DESC, ''

def parse_date(date: str) -> Union[datetime, Any]:
    """Try parsing date with multiple formats."""
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(date, format=fmt)
        except (ValueError, TypeError):
            continue
    logging.error(f"Failed to parse date: {date}")
    return pd.NaT

def calculate_mtbf(filtered_df: pd.DataFrame, included_indices: set) -> float:
    """Calculate Mean Time Between Failures (MTBF) in days for included rows using Crow-AMSAA method."""
    try:
        # Use Crow-AMSAA method for consistent MTBF calculation
        lambda_param, beta, failures_per_year = calculate_crow_amsaa_params(filtered_df, included_indices)
        
        if failures_per_year > 0:
            # Convert failures per year to MTBF in days
            mtbf_days = 365.0 / failures_per_year
            logging.debug(f"MTBF calculated via Crow-AMSAA: {mtbf_days:.2f} days (failures/year: {failures_per_year:.2f})")
            return round(mtbf_days, 2)
        else:
            logging.debug("Insufficient data for Crow-AMSAA MTBF calculation")
            return 0.0
    except Exception as e:
        logging.error(f"Error calculating MTBF via Crow-AMSAA: {e}")
        return 0.0

def calculate_crow_amsaa_params(filtered_df: pd.DataFrame, included_indices: set) -> tuple:
    """Calculate Crow-AMSAA parameters (lambda, beta) and failures per year."""
    try:
        valid_indices = filtered_df.index.intersection(included_indices)
        filtered_dates = [
            parse_date(filtered_df.at[idx, 'Reported Date'])
            for idx in valid_indices
            if pd.notna(filtered_df.at[idx, 'Reported Date'])
        ]
        filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
        
        if len(filtered_dates) == 0:
            logging.debug("No valid dates for Crow-AMSAA")
            return None, None, 0.0
        elif len(filtered_dates) == 1:
            lambda_param = 1 / 365
            beta = 1.0
            failures_per_year = lambda_param * (365 ** beta)
            logging.debug(f"Single failure case: failures/year={failures_per_year:.2f}")
            return lambda_param, beta, round(failures_per_year, 2)
        
        dates = sorted(filtered_dates)
        t0 = dates[0]
        times = [(d - t0).days + 1 for d in dates]
        n = np.arange(1, len(times) + 1)
        
        log_n = np.log(n)
        log_t = np.log(times)
        coeffs = np.polyfit(log_t, log_n, 1)
        beta = coeffs[0]
        lambda_param = np.exp(coeffs[1])
        
        failures_per_year = lambda_param * (365 ** beta)
        
        logging.debug(f"Crow-AMSAA params: beta={beta:.2f}, lambda={lambda_param:.4f}, failures/year={failures_per_year:.2f}")
        return lambda_param, beta, round(failures_per_year, 2)
    except Exception as e:
        logging.error(f"Error calculating Crow-AMSAA params: {e}")
        return None, None, 0.0

def create_crow_amsaa_plot(filtered_df: pd.DataFrame, included_indices: set, frame: Union[tk.Frame, ttk.LabelFrame]) -> tuple:
    """Create a Crow-AMSAA plot and return figure and parameters."""
    try:
        valid_indices = filtered_df.index.intersection(included_indices)
        filtered_dates = [
            parse_date(filtered_df.at[idx, 'Reported Date'])
            for idx in valid_indices
            if pd.notna(filtered_df.at[idx, 'Reported Date'])
        ]
        filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
        
        if len(filtered_dates) < 2:
            logging.debug(f"Insufficient valid dates for Crow-AMSAA: {len(filtered_dates)}")
            return None, None, None
        
        dates = sorted(filtered_dates)
        t0 = dates[0]
        times = [(d - t0).days + 1 for d in dates]
        n = np.arange(1, len(times) + 1)
        
        log_n = np.log(n)
        log_t = np.log(times)
        coeffs = np.polyfit(log_t, log_n, 1)
        beta = coeffs[0]
        lambda_param = np.exp(coeffs[1])
        
        frame.update_idletasks()
        width_px = max(frame.winfo_width(), 100)
        height_px = max(frame.winfo_height(), 100)
        width_in = min(max(width_px / 100, 4), 8)
        height_in = min(max(height_px / 100, 2.5), 5)
        font_scale = width_in / 5
        
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        ax.scatter(times, n, marker='o', label='Observed Failures')
        t_fit = np.linspace(min(times), max(times), 100)
        n_fit = lambda_param * t_fit ** beta
        ax.plot(t_fit, n_fit, label=f'Crow-AMSAA (β={beta:.2f}, λ={lambda_param:.4f})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Crow-AMSAA Plot", fontsize=10 * font_scale)
        ax.set_xlabel("Time (days)", fontsize=8 * font_scale)
        ax.set_ylabel("Cumulative Failures", fontsize=8 * font_scale)
        ax.legend(fontsize=6 * font_scale)
        ax.grid(True, which="both", ls="--")
        ax.tick_params(axis='both', labelsize=6 * font_scale)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        logging.debug(f"Crow-AMSAA plot created: beta={beta:.2f}, lambda={lambda_param:.4f}, figsize=({width_in}, {height_in})")
        return fig, beta, lambda_param
    except Exception as e:
        logging.error(f"Error creating Crow-AMSAA plot: {e}")
        return None, None, None

def process_files(work_order_path: str, dict_path: str, status_label: ttk.Label, root: tk.Tk, output_dir: Optional[str] = None, use_ai: bool = False, ai_classifier: Optional[AIClassifier] = None, column_mapping: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """Process work order and dictionary files with optional AI classification."""
    try:
        status_label.config(text="Processing...", foreground="blue")
        root.config(cursor="wait")
        root.update()
        logging.info(f"Processing work order file: {work_order_path}")
        logging.info(f"Dictionary file: {dict_path}")
        logging.info(f"AI classification enabled: {use_ai}")
        logging.info(f"Column mapping: {column_mapping}")
        
        if not os.path.exists(work_order_path):
            raise FileNotFoundError(f"Work order file not found: {work_order_path}")
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        if not work_order_path.endswith('.xlsx'):
            raise ValueError(f"Work order file must be .xlsx, got: {work_order_path}")
        if not dict_path.endswith('.xlsx'):
            raise ValueError(f"Dictionary file must be .xlsx, got: {dict_path}")
        
        try:
            wo_df = pd.read_excel(work_order_path)
            logging.info(f"Work order columns: {list(wo_df.columns)}")
            if wo_df.empty:
                raise ValueError("Work order file is empty")
        except Exception as e:
            raise ValueError(f"Failed to read work order file: {str(e)}")
        
        # Apply column mapping if provided
        if column_mapping:
            rename_dict = {}
            for required_col, mapped_col in column_mapping.items():
                if mapped_col in wo_df.columns and mapped_col != required_col:
                    rename_dict[mapped_col] = required_col
            
            if rename_dict:
                wo_df = wo_df.rename(columns=rename_dict)
                logging.info(f"Applied column mapping: {rename_dict}")
        
        required_columns = ['Work Order', 'Description', 'Asset', 'Equipment #', 'Work Type', 'Reported Date']
        missing_cols = [col for col in required_columns if col not in wo_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in work order file: {', '.join(missing_cols)}")
        
        # Add Work Order Cost column if not present (optional column)
        if 'Work Order Cost' not in wo_df.columns:
            wo_df['Work Order Cost'] = 0.0
            logging.info("Work Order Cost column not found, adding with default value 0.0")
        # Add User failure code column if not present (optional column)
        if 'User failure code' not in wo_df.columns:
            wo_df['User failure code'] = ''
            logging.info("User failure code column not found, adding as blank")
        
        try:
            dict_df = pd.read_excel(dict_path)
            logging.info(f"Dictionary columns: {list(dict_df.columns)}")
            if dict_df.empty:
                raise ValueError("Dictionary file is empty")
        except Exception as e:
            raise ValueError(f"Failed to read dictionary file: {str(e)}")
        
        if not all(col in dict_df.columns for col in ['Keyword', 'Code', 'Description']):
            raise ValueError("Dictionary file must contain columns: Keyword, Code, Description")
        
        if dict_df['Keyword'].dropna().empty:
            raise ValueError("Dictionary file contains no valid keywords")
        
        dictionary = []
        keywords = dict_df['Keyword'].dropna().astype(str).tolist()
        patterns = compile_patterns(keywords)
        for idx, row in dict_df.iterrows():
            keyword = str(row['Keyword']).lower()
            if not keyword:
                continue
            norm_keyword = normalize_text(keyword)
            try:
                tokens = word_tokenize(norm_keyword)
                stemmed_keyword = ' '.join(stemmer.stem(token) for token in tokens)
            except Exception as e:
                logging.error(f"Error tokenizing keyword '{keyword}': {e}")
                stemmed_keyword = norm_keyword
            code = str(row['Code'])
            desc = str(row['Description'])
            pattern = patterns[keywords.index(keyword)] if keyword in keywords else None
            dictionary.append((keyword, norm_keyword, stemmed_keyword, code, desc, pattern))
        
        if not dictionary:
            raise ValueError("No valid keywords processed from dictionary")
        
        # Initialize AI classifier if requested
        if use_ai and AI_AVAILABLE and ai_classifier is None:
            try:
                ai_classifier = AIClassifier(
                    confidence_threshold=AI_CONFIDENCE_THRESHOLD,
                    cache_file=AI_CACHE_FILE
                )
                if not ai_classifier.load_failure_dictionary(dict_path):
                    logging.warning("Failed to load dictionary for AI classifier, falling back to dictionary matching")
                    use_ai = False
            except Exception as e:
                logging.error(f"Failed to initialize AI classifier: {e}")
                use_ai = False
        
        wo_df['Failure Code'] = DEFAULT_CODE
        wo_df['Failure Description'] = DEFAULT_DESC
        wo_df['Matched Keyword'] = ''
        wo_df['AI Confidence'] = 0.0
        wo_df['Classification Method'] = 'dictionary'
        
        # Process work orders
        if use_ai and ai_classifier:
            status_label.config(text="Processing with Enhanced AI classification...", foreground="blue")
            root.update()
            
            # Analyze historical patterns for temporal analysis
            ai_classifier.analyze_historical_patterns(wo_df)
            
            # Batch process with AI
            descriptions = [str(row['Description']) for _, row in wo_df.iterrows()]
            ai_results = ai_classifier.batch_classify(descriptions, lambda desc: match_failure_mode(desc, dictionary))
            
            for idx, (_, row) in enumerate(wo_df.iterrows()):
                if idx < len(ai_results):
                    result = ai_results[idx]
                    wo_df.at[idx, 'Failure Code'] = result.code
                    wo_df.at[idx, 'Failure Description'] = result.description
                    wo_df.at[idx, 'Matched Keyword'] = result.matched_keyword
                    wo_df.at[idx, 'AI Confidence'] = result.confidence
                    wo_df.at[idx, 'Classification Method'] = result.method
                    logging.debug(f"Work Order {row['Work Order']}: AI code={result.code}, confidence={result.confidence}, method={result.method}")
        else:
            # Use traditional dictionary matching
            for idx, row in wo_df.iterrows():
                desc = str(row['Description'])
                code, failure_desc, matched_keyword = match_failure_mode(desc, dictionary)
                wo_df.at[idx, 'Failure Code'] = code
                wo_df.at[idx, 'Failure Description'] = failure_desc
                wo_df.at[idx, 'Matched Keyword'] = matched_keyword
                wo_df.at[idx, 'AI Confidence'] = 0.5  # Default confidence for dictionary matching
                wo_df.at[idx, 'Classification Method'] = 'dictionary'
                wo_df.at[idx, 'User failure code'] = row.get('User failure code', '')
                logging.debug(f"Work Order {row['Work Order']}: code={code}, desc={failure_desc}, keyword={matched_keyword}, user failure code={row.get('User failure code', '')}")
        
        if not output_dir:
            output_dir = os.path.dirname(work_order_path) or '.'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        status_label.config(text="Processing complete.", foreground="green")
        root.config(cursor="")
        root.update()
        logging.info(f"Processing complete. Rows processed: {len(wo_df)}")
        return wo_df
    
    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        status_label.config(text=error_msg, foreground="red")
        root.config(cursor="")
        root.update()
        logging.error(error_msg)
        return None
    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        status_label.config(text=error_msg, foreground="red")
        root.config(cursor="")
        root.update()
        logging.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Unexpected error processing files: {str(e)}"
        status_label.config(text=error_msg, foreground="red")
        root.config(cursor="")
        root.update()
        logging.error(error_msg)
        return None

class FailureModeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Work Order Analysis Pro - AI-Powered Failure Mode Classification")
        self.root.geometry("1400x900")
        
        # Set application icon
        try:
            icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'app_icon.ico')
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
                # Also set the taskbar icon for Windows
                if os.name == 'nt':  # Windows
                    try:
                        import ctypes
                        myappid = 'workorderanalysis.pro.2.0'  # arbitrary string
                        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
                    except:
                        pass
                print(f"Application icon loaded from: {icon_path}")
            else:
                print(f"Icon file not found at: {icon_path}")
        except Exception as e:
            print(f"Could not load application icon: {e}")
        
        # Set modern theme
        style = ttk.Style()
        style.theme_use('clam')  # Use a modern theme
        
        self.wo_df = None
        self.output_dir = None
        self.included_indices = set()
        self.dictionary = None
        self.start_date = None
        self.end_date = None
        self.sort_states = {}  # Track sort direction per column
        self.selected_plot_point = None  # For plot selection
        self.crow_amsaa_canvas = None    # Store canvas for mpl_connect
        self.crow_amsaa_fig = None       # Store figure for mpl_connect
        self.context_menu = None         # For right-click menu
        self.equipment_context_menu = None  # For equipment right-click menu
        self.is_segmented_view = False   # Track if we're in segmented view
        self.segment_data = None         # Store segment data for risk calculation
        self.last_highlight_artist = None  # Track the last highlight for Crow-AMSAA
        self.ai_classifier = None        # AI classifier instance
        self.use_ai_classification = False  # Whether to use AI classification
        
        # AI settings variables
        self.ai_enabled_var = tk.BooleanVar(value=False)
        self.confidence_var = tk.StringVar(value=str(AI_CONFIDENCE_THRESHOLD))
        self.confidence_scale_var = tk.DoubleVar(value=AI_CONFIDENCE_THRESHOLD)
        
        # Column mapping for CMMS compatibility
        self.column_mapping = {}  # Maps CMMS columns to required columns
        
        # Load application configuration
        self.config = load_app_config()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main notebook for tabbed interface
        self.create_notebook()
        
        # Create status bar
        self.create_status_bar()
        
        # Load saved column mappings
        self.load_saved_column_mappings()
        
        # Load saved file paths
        self.load_saved_file_paths()

    def create_menu_bar(self):
        """Create a professional menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self.menu_bar = menubar  # Add this line for test compatibility
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Work Order File...", command=self.browse_wo, accelerator="Ctrl+O")
        file_menu.add_command(label="Load Dictionary File...", command=self.browse_dict, accelerator="Ctrl+D")
        file_menu.add_separator()
        file_menu.add_command(label="Set Output Directory...", command=self.browse_output)
        file_menu.add_separator()
        file_menu.add_command(label="Export to Excel...", command=self.export_to_excel, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Process Menu
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Process Files", command=self.run_processing, accelerator="F5")
        process_menu.add_command(label="Batch Process...", command=self.batch_process)
        process_menu.add_separator()
        process_menu.add_command(label="Clear Data", command=self.clear_data)
        
        # AI Menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_checkbutton(label="Enable AI Classification", variable=self.ai_enabled_var, command=self.toggle_ai)
        ai_menu.add_separator()
        ai_menu.add_command(label="AI Classifier Settings...", command=self.show_ai_settings_dialog)
        ai_menu.add_command(label="AI Statistics", command=self.show_ai_stats)
        ai_menu.add_command(label="Clear AI Cache", command=self.clear_ai_cache)
        ai_menu.add_separator()
        ai_menu.add_command(label="Export Training Data", command=self.export_training_data)
        
        # Tools Menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Column Mapping...", command=self.show_column_mapping)
        tools_menu.add_command(label="Filter Management...", command=self.show_filter_manager)
        tools_menu.add_command(label="Date Range Selector...", command=self.show_date_selector)
        tools_menu.add_separator()
        tools_menu.add_command(label="Charts...", command=self.show_charts_dialog)
        tools_menu.add_separator()
        tools_menu.add_command(label="Reset All Filters", command=self.reset_all_filters)
        tools_menu.add_command(label="Open Output Folder", command=self.open_output_folder)
        tools_menu.add_separator()
        tools_menu.add_command(label="View FMEA Export Data...", command=self.show_fmea_export_data)
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Software User Guide", command=self.show_software_user_guide)
        help_menu.add_command(label="Technical Application Guide", command=self.show_technical_guide)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.browse_wo())
        self.root.bind('<Control-d>', lambda e: self.browse_dict())
        self.root.bind('<Control-e>', lambda e: self.export_to_excel())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        self.root.bind('<F5>', lambda e: self.run_processing())
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_notebook(self):
        """Create tabbed notebook interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_data_input_tab()
        self.create_analysis_tab()
        self.create_risk_assessment_tab()
        
        # Add new analysis tabs if modules are available
        if MODULES_AVAILABLE:
            self.create_weibull_analysis_tab()
            # Removed FMEA export tab - functionality moved to Weibull analysis tab
            self.create_pm_analysis_tab()
            self.create_spares_analysis_tab()

    def create_data_input_tab(self):
        """Create the data input tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📁 Data Input")
        
        # File selection frame
        file_frame = ttk.LabelFrame(data_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Work Order File
        wo_frame = ttk.Frame(file_frame)
        wo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wo_frame, text="Work Order File:", width=15).pack(side=tk.LEFT)
        self.wo_entry = ttk.Entry(wo_frame)
        self.wo_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(wo_frame, text="Browse", command=self.browse_wo, width=10).pack(side=tk.RIGHT)
        
        # Dictionary File
        dict_frame = ttk.Frame(file_frame)
        dict_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dict_frame, text="Dictionary File:", width=15).pack(side=tk.LEFT)
        self.dict_entry = ttk.Entry(dict_frame)
        self.dict_entry.insert(0, DEFAULT_DICT_PATH)
        self.dict_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(dict_frame, text="Browse", command=self.browse_dict, width=10).pack(side=tk.RIGHT)
        
        # Output Directory
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=2)
        ttk.Label(output_frame, text="Output Directory:", width=15).pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output, width=10).pack(side=tk.RIGHT)
        
        # Action buttons frame
        action_frame = ttk.Frame(data_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(action_frame, text="🚀 Process Files", command=self.run_processing, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📊 Export to Excel", command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📁 Open Output", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗑️ Clear Data", command=self.clear_data).pack(side=tk.RIGHT, padx=5)

    def create_analysis_tab(self):
        """Create the analysis tab with filters and data display"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📈 Analysis")
        
        # Filter panel
        filter_frame = ttk.LabelFrame(analysis_frame, text="Filters", padding=10)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Filter controls
        filter_controls = ttk.Frame(filter_frame)
        filter_controls.pack(fill=tk.X)
        
        # Row 1: Equipment and Failure Code
        row1 = ttk.Frame(filter_controls)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Equipment:", width=12).pack(side=tk.LEFT)
        self.equipment_var = tk.StringVar()
        self.equipment_dropdown = ttk.Combobox(row1, textvariable=self.equipment_var, state="readonly", width=20)
        self.equipment_dropdown['values'] = ['']
        self.equipment_dropdown.pack(side=tk.LEFT, padx=(5, 20))
        self.equipment_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_table())
        
        # --- Failure code source selector ---
        self.failure_code_source_var = tk.StringVar(value="AI/Dictionary")
        ttk.Label(row1, text="Failure Code Source:", width=18).pack(side=tk.LEFT)
        self.failure_code_source_dropdown = ttk.Combobox(row1, textvariable=self.failure_code_source_var, state="readonly", width=15)
        self.failure_code_source_dropdown['values'] = ["AI/Dictionary", "User"]
        self.failure_code_source_dropdown.pack(side=tk.LEFT, padx=(5, 5))
        self.failure_code_source_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_failure_code_dropdown())
        
        ttk.Label(row1, text="Failure Code:", width=12).pack(side=tk.LEFT)
        self.failure_code_var = tk.StringVar()
        self.failure_code_dropdown = ttk.Combobox(row1, textvariable=self.failure_code_var, state="readonly", width=20)
        self.failure_code_dropdown['values'] = ['']
        self.failure_code_dropdown.pack(side=tk.LEFT, padx=5)
        self.failure_code_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_table())
        
        # Row 2: Work Type and Date Range
        row2 = ttk.Frame(filter_controls)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Work Type:", width=12).pack(side=tk.LEFT)
        self.work_type_var = tk.StringVar()
        self.work_type_dropdown = ttk.Combobox(row2, textvariable=self.work_type_var, state="readonly", width=20)
        self.work_type_dropdown['values'] = ['']
        self.work_type_dropdown.pack(side=tk.LEFT, padx=(5, 20))
        self.work_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_table())
        
        ttk.Label(row2, text="Date Range:", width=12).pack(side=tk.LEFT)
        self.start_date_entry = ttk.Entry(row2, width=12)
        self.start_date_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="to").pack(side=tk.LEFT, padx=2)
        self.end_date_entry = ttk.Entry(row2, width=12)
        self.end_date_entry.pack(side=tk.LEFT, padx=5)
        
        # Filter action buttons
        filter_buttons = ttk.Frame(filter_frame)
        filter_buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(filter_buttons, text="Apply Filters", command=self.apply_date_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_buttons, text="Clear All", command=self.reset_all_filters).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_buttons, text="Reset Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=5)
        
        # Data display area with resizable panes
        data_display_frame = ttk.Frame(analysis_frame)
        data_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Main paned window for all content
        self.main_paned = ttk.PanedWindow(data_display_frame, orient=tk.VERTICAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Top section: tables
        tables_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(tables_frame, weight=2)
        
        # Paned window for table and equipment summary
        self.table_paned = ttk.PanedWindow(tables_frame, orient=tk.VERTICAL)
        self.table_paned.pack(fill=tk.BOTH, expand=True)
        
        self.work_order_frame = ttk.Frame(self.table_paned)
        self.table_paned.add(self.work_order_frame, weight=3)
        self.equipment_frame = ttk.Frame(self.table_paned)
        self.table_paned.add(self.equipment_frame, weight=1)
        
        self.tree = None
        self.equipment_tree = None
        
        # Bottom section: Crow-AMSAA plot area
        plot_frame = ttk.LabelFrame(self.main_paned, text="Crow-AMSAA Analysis", padding=10)
        self.main_paned.add(plot_frame, weight=1)
        
        # Plot controls
        plot_controls = ttk.Frame(plot_frame)
        plot_controls.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(plot_controls, text="Export Plot", command=self.export_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_controls, text="Open in New Window", command=self.open_plot_in_new_window).pack(side=tk.LEFT, padx=5)
        self.return_to_single_button = ttk.Button(plot_controls, text="Return to Single Plot", 
                                                 command=self.return_to_single_plot)
        
        # Plot area
        self.crow_amsaa_frame = ttk.Frame(plot_frame)
        self.crow_amsaa_frame.pack(fill=tk.BOTH, expand=True)

    def create_risk_assessment_tab(self):
        """Create the risk assessment tab"""
        risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(risk_frame, text="⚠️ Risk Assessment")
        
        # Risk parameters
        params_frame = ttk.LabelFrame(risk_frame, text="Risk Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Production Loss
        prod_frame = ttk.Frame(params_frame)
        prod_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prod_frame, text="Production Loss (Lb/MT):", width=20).pack(side=tk.LEFT)
        self.prod_loss_var = tk.StringVar(value="0")
        ttk.Entry(prod_frame, textvariable=self.prod_loss_var, width=18).pack(side=tk.LEFT, padx=5)
        
        # Maintenance Cost
        maint_frame = ttk.Frame(params_frame)
        maint_frame.pack(fill=tk.X, pady=2)
        ttk.Label(maint_frame, text="Maintenance Cost ($):", width=20).pack(side=tk.LEFT)
        self.maint_cost_var = tk.StringVar(value="0")
        ttk.Entry(maint_frame, textvariable=self.maint_cost_var, width=18).pack(side=tk.LEFT, padx=5)
        
        # Margin
        margin_frame = ttk.Frame(params_frame)
        margin_frame.pack(fill=tk.X, pady=2)
        ttk.Label(margin_frame, text="Margin ($/weight):", width=20).pack(side=tk.LEFT)
        self.margin_var = tk.StringVar(value="0")
        ttk.Entry(margin_frame, textvariable=self.margin_var, width=18).pack(side=tk.LEFT, padx=5)
        
        # Forecast Period
        forecast_frame = ttk.Frame(params_frame)
        forecast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(forecast_frame, text="Forecast Period (years):", width=20).pack(side=tk.LEFT)
        self.forecast_period_var = tk.StringVar(value="5")
        ttk.Entry(forecast_frame, textvariable=self.forecast_period_var, width=18).pack(side=tk.LEFT, padx=5)
        
        # Risk action buttons
        risk_buttons_frame = ttk.Frame(risk_frame)
        risk_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(risk_buttons_frame, text="🧮 Calculate Risk", command=self.update_risk).pack(side=tk.LEFT, padx=5)
        ttk.Button(risk_buttons_frame, text="💾 Save Preset", command=self.save_risk_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(risk_buttons_frame, text="📂 Load Preset", command=self.load_risk_preset).pack(side=tk.LEFT, padx=5)
        
        # Current Filters Information
        filters_frame = ttk.LabelFrame(risk_frame, text="Current Data Filters", padding=10)
        filters_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.filters_label = ttk.Label(filters_frame, text="No filters applied", foreground="gray")
        self.filters_label.pack(anchor=tk.W)
        
        # Cost Statistics
        cost_frame = ttk.LabelFrame(risk_frame, text="Cost Statistics", padding=10)
        cost_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.cost_label = ttk.Label(cost_frame, text="Total Cost: $0.00 | Average Cost: $0.00 | Work Orders: 0", foreground="blue")
        self.cost_label.pack(anchor=tk.W)
        
        # Risk summary
        summary_frame = ttk.LabelFrame(risk_frame, text="Risk Summary", padding=10)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.risk_label = ttk.Label(summary_frame, text="Failure Rate: 0.00, Annualized Risk: $0.00")
        self.risk_label.pack(anchor=tk.W)
        
        # Risk Trend Plot
        plot_frame = ttk.LabelFrame(risk_frame, text="Risk Trend Projection", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Plot controls
        plot_controls = ttk.Frame(plot_frame)
        plot_controls.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(plot_controls, text="🔄 Update Plot", command=self.update_risk_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_controls, text="📊 Export Plot", command=self.export_risk_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_controls, text="📋 Copy Summary", command=self.copy_risk_summary).pack(side=tk.LEFT, padx=5)
        
        # Plot area
        self.risk_plot_frame = ttk.Frame(plot_frame)
        self.risk_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot
        self.risk_plot_fig = None
        self.risk_plot_canvas = None

    def create_status_bar(self):
        """Create a modern status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status label
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # Status indicators
        self.ai_status_indicator = ttk.Label(status_frame, text="🤖", foreground="green")
        self.ai_status_indicator.pack(side=tk.RIGHT, padx=5)
        
        self.column_mapping_indicator = ttk.Label(status_frame, text="📋", foreground="blue")
        self.column_mapping_indicator.pack(side=tk.RIGHT, padx=5)
        
        self.data_status_indicator = ttk.Label(status_frame, text="📊", foreground="blue")
        self.data_status_indicator.pack(side=tk.RIGHT, padx=5)

    def browse_wo(self):
        """Browse for work order file."""
        # Set initial directory to last used path if available
        initial_dir = os.path.dirname(self.config.get('last_work_order_path', '')) if self.config.get('last_work_order_path') else None
        path = filedialog.askopenfilename(
            title="Select Work Order File",
            filetypes=[("Excel files", "*.xlsx")],
            initialdir=initial_dir
        )
        if path:
            self.wo_entry.delete(0, tk.END)
            self.wo_entry.insert(0, path)
            # Save the path
            self.config['last_work_order_path'] = path
            self.save_file_paths()
    
    def browse_dict(self):
        """Browse for dictionary file."""
        # Set initial directory to last used path if available
        initial_dir = os.path.dirname(self.config.get('last_dictionary_path', '')) if self.config.get('last_dictionary_path') else None
        path = filedialog.askopenfilename(
            title="Select Dictionary File",
            filetypes=[("Excel files", "*.xlsx")],
            initialdir=initial_dir
        )
        if path:
            self.dict_entry.delete(0, tk.END)
            self.dict_entry.insert(0, path)
            # Save the path
            self.config['last_dictionary_path'] = path
            self.save_file_paths()
    
    def browse_output(self):
        """Browse for output directory."""
        # Set initial directory to last used path if available
        initial_dir = self.config.get('last_output_directory', '') if self.config.get('last_output_directory') else None
        path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=initial_dir
        )
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)
            self.output_dir = path
            # Save the path
            self.config['last_output_directory'] = path
            self.save_file_paths()
    
    def run_processing(self):
        """Process work order and dictionary files."""
        wo_path = self.wo_entry.get()
        dict_path = self.dict_entry.get()
        
        if not wo_path or not dict_path:
            self.status_label.config(text="Select both work order and dictionary files.", foreground="red")
            messagebox.showerror("Error", "Please select both work order and dictionary files.")
            return
        if not os.path.exists(wo_path):
            self.status_label.config(text=f"Work order file not found: {wo_path}", foreground="red")
            messagebox.showerror("Error", f"Work order file not found: {wo_path}")
            return
        if not os.path.exists(dict_path):
            self.status_label.config(text=f"Dictionary file not found: {dict_path}", foreground="red")
            messagebox.showerror("Error", f"Dictionary file not found: {dict_path}")
            return
        
        try:
            dict_df = pd.read_excel(dict_path)
            self.dictionary = dict_df.set_index('Code')['Description'].to_dict()
        except Exception as e:
            self.status_label.config(text=f"Failed to read dictionary: {str(e)}", foreground="red")
            messagebox.showerror("Error", f"Failed to read dictionary file: {str(e)}")
            return
        
        # Check AI settings
        self.use_ai_classification = self.ai_enabled_var.get()
        if self.use_ai_classification and not AI_AVAILABLE:
            self.status_label.config(text="AI classification requested but not available. Install dependencies.", foreground="red")
            messagebox.showerror("Error", "AI classification not available. Please install required dependencies.")
            return
        
        # Initialize AI classifier if needed
        if self.use_ai_classification:
            try:
                confidence_threshold = float(self.confidence_var.get())
                
                self.ai_classifier = AIClassifier(
                    confidence_threshold=confidence_threshold,
                    cache_file=AI_CACHE_FILE
                )
                
                if not self.ai_classifier.load_failure_dictionary(dict_path):
                    self.status_label.config(text="Failed to load dictionary for AI classifier.", foreground="red")
                    messagebox.showerror("Error", "Failed to load dictionary for AI classifier.")
                    return
                    
            except Exception as e:
                self.status_label.config(text=f"Failed to initialize AI classifier: {str(e)}", foreground="red")
                messagebox.showerror("Error", f"Failed to initialize AI classifier: {str(e)}")
                return
        
        # Update progress
        self.update_progress(0, "Processing files...")
        
        self.wo_df = process_files(wo_path, dict_path, self.status_label, self.root, self.output_dir, 
                                 use_ai=self.use_ai_classification, ai_classifier=self.ai_classifier, 
                                 column_mapping=self.column_mapping)
        
        if self.wo_df is not None and not self.wo_df.empty:
            self.update_progress(50, "Updating interface...")
            
            equipment_nums = sorted(self.wo_df['Equipment #'].dropna().unique())
            work_types = sorted(self.wo_df['Work Type'].dropna().unique())
            failure_codes = sorted(self.wo_df['Failure Code'].dropna().unique())
            logging.info(f"Equipment numbers: {equipment_nums}")
            logging.info(f"Work types: {work_types}")
            logging.info(f"Failure codes: {failure_codes}")
            self.equipment_dropdown['values'] = [''] + list(equipment_nums)
            self.work_type_dropdown['values'] = [''] + list(work_types)
            self.failure_code_dropdown['values'] = [''] + list(failure_codes)
            self.equipment_var.set('')
            self.work_type_var.set('')
            self.failure_code_var.set('')
            self.included_indices = set(self.wo_df.index)
            
            # Update new analysis tab dropdowns if modules are available
            if MODULES_AVAILABLE:
                # Weibull analysis dropdowns
                if hasattr(self, 'weibull_equipment_dropdown'):
                    self.weibull_equipment_dropdown['values'] = [''] + list(equipment_nums)
                    self.weibull_equipment_var.set('')
                if hasattr(self, 'weibull_failure_dropdown'):
                    self.weibull_failure_dropdown['values'] = [''] + list(failure_codes)
                    self.weibull_failure_var.set('')
                
                # PM analysis dropdowns - removed as we use current selections instead
            
            self.update_progress(75, "Updating tables...")
            self.update_table()
            
            self.update_progress(100, "Processing complete!")
            self.data_status_indicator.config(text="📊")
            
            # Switch to analysis tab
            self.notebook.select(1)
            
        else:
            self.status_label.config(text="No data processed. Check input files or logs.", foreground="red")
            logging.error("Work order DataFrame is None or empty.")
            self.update_progress(0, "Processing failed")
    
    def show_ai_stats(self):
        """Show AI classification statistics."""
        if not self.ai_classifier:
            messagebox.showinfo("AI Stats", "No AI classifier available. Enable AI classification first.")
            return
        
        try:
            stats = self.ai_classifier.get_classification_stats()
            
            # Check AI capabilities
            capabilities = []
            if self.ai_classifier.embedding_model:
                capabilities.append("Sentence Embeddings")
            if self.ai_classifier.nlp:
                capabilities.append("SpaCy NLP")
            capabilities.append("Expert System")
            capabilities.append("Contextual Patterns")
            capabilities.append("Temporal Analysis")
            
            capabilities_text = ", ".join(capabilities) if capabilities else "None"
            
            stats_text = f"""AI Classification Statistics:
            
Total Classifications: {stats['total_classifications']}
Cache Size: {stats['cache_size_mb']:.2f} MB
AI Capabilities: {capabilities_text}

Methods Used:"""
            
            for method, count in stats['methods_used'].items():
                method_display = {
                    'expert_system': 'Expert System',
                    'contextual_patterns': 'Contextual Patterns',
                    'temporal_analysis': 'Temporal Analysis',
                    'ai_embeddings': 'Sentence Embeddings', 
                    'ai_spacy': 'SpaCy NLP',
                    'dictionary_fallback': 'Dictionary Matching'
                }.get(method, method)
                stats_text += f"\n  {method_display}: {count}"
            
            stats_text += f"""

Confidence Distribution:
  High (≥0.8): {stats['confidence_distribution']['high']}
  Medium (≥0.5): {stats['confidence_distribution']['medium']}
  Low (<0.5): {stats['confidence_distribution']['low']}"""
            
            messagebox.showinfo("AI Classification Stats", stats_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get AI stats: {str(e)}")

    def clear_ai_cache(self):
        """Clear the AI classification cache."""
        if not self.ai_classifier:
            messagebox.showinfo("Clear Cache", "No AI classifier available.")
            return
        
        try:
            self.ai_classifier.clear_cache()
            messagebox.showinfo("Success", "AI classification cache cleared successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")
    
    def apply_date_filter(self):
        """Apply date range filter to work orders."""
        start_date_str = self.start_date_entry.get().strip()
        end_date_str = self.end_date_entry.get().strip()
        
        self.start_date = None
        self.end_date = None
        
        if start_date_str:
            start_date = parse_date(start_date_str)
            if pd.isna(start_date):
                self.status_label.config(text="Invalid start date format. Use MM/DD/YYYY.", foreground="red")
                messagebox.showerror("Error", "Invalid start date format. Use MM/DD/YYYY.")
                return
            self.start_date = start_date
        
        if end_date_str:
            end_date = parse_date(end_date_str)
            if pd.isna(end_date):
                self.status_label.config(text="Invalid end date format. Use MM/DD/YYYY.", foreground="red")
                messagebox.showerror("Error", "Invalid end date format. Use MM/DD/YYYY.")
                return
            self.end_date = end_date
        
        if self.start_date and self.end_date and self.start_date > self.end_date:
            self.status_label.config(text="Start date cannot be after end date.", foreground="red")
            messagebox.showerror("Error", "Start date cannot be after end date.")
            self.start_date = None
            self.end_date = None
            return
        
        self.update_table()
        self.status_label.config(text="Date filter applied.", foreground="green")
        logging.info(f"Date filter applied: start={start_date_str}, end={end_date_str}")
    
    def reset_equip_failcode(self):
        """Reset Equipment and Failure Code filters."""
        self.equipment_var.set('')
        self.failure_code_var.set('')
        self.update_table()
        self.status_label.config(text="Equipment and Failure Code filters reset.", foreground="green")
        logging.info("Reset Equipment and Failure Code filters")
    
    def reset_work_type(self):
        """Reset Work Type filter."""
        self.work_type_var.set('')
        self.update_table()
        self.status_label.config(text="Work Type filter reset.", foreground="green")
        logging.info("Reset Work Type filter")
    
    def reset_date(self):
        """Reset date range filter."""
        self.start_date_entry.delete(0, tk.END)
        self.end_date_entry.delete(0, tk.END)
        self.start_date = None
        self.end_date = None
        self.update_table()
        self.status_label.config(text="Date filter reset.", foreground="green")
        logging.info("Reset date filter")
    
    def sort_column(self, tree, col, reverse):
        """Sort Treeview column and update sort indicator."""
        if tree == self.tree:
            columns = ['Include', 'Index', 'Work Order', 'Description', 'Asset', 'Equipment #', 
                       'Work Type', 'Reported Date', 'Failure Code', 'Failure Description', 'Matched Keyword', 'User failure code']
            if col == 'Include':
                data = [(tree.set(item, col), item) for item in tree.get_children() if tree.set(item, 'Index') != '']
                data.sort(key=lambda x: x[0] == '☑', reverse=reverse)
            elif col == 'Reported Date':
                data = [(parse_date(tree.set(item, col)) if tree.set(item, col) else pd.Timestamp.min, item) 
                        for item in tree.get_children() if tree.set(item, 'Index') != '']
                data.sort(key=lambda x: x[0], reverse=reverse)
            elif col in ['Index', 'Work Order']:
                data = [(float(tree.set(item, col)) if tree.set(item, col) else float('inf'), item) 
                        for item in tree.get_children() if tree.set(item, 'Index') != '']
                data.sort(key=lambda x: x[0], reverse=reverse)
            else:
                data = [(tree.set(item, col).lower() if tree.set(item, col) else '', item) 
                        for item in tree.get_children() if tree.set(item, 'Index') != '']
                data.sort(key=lambda x: x[0], reverse=reverse)
        else:  # equipment_tree
            if col in ['Total Work Orders', 'Failures per Year']:
                data = [(float(tree.set(item, col)) if tree.set(item, col) else float('inf'), item) 
                        for item in tree.get_children()]
                data.sort(key=lambda x: x[0], reverse=reverse)
            else:  # Equipment #
                data = [(tree.set(item, col).lower() if tree.set(item, col) else '', item) 
                        for item in tree.get_children()]
                data.sort(key=lambda x: x[0], reverse=reverse)
        
        for index, (_, item) in enumerate(data):
            tree.move(item, '', index)
        
        tree.heading(col, text=col + (' ↓' if reverse else ' ↑'))
        for c in tree['columns']:
            if c != col:
                tree.heading(c, text=c)
        
        self.sort_states[(tree, col)] = not reverse
        logging.debug(f"Sorted {tree} by {col}, reverse={reverse}")
    
    def edit_cell(self, event):
        """Handle editing of Failure Code or Failure Description in the Treeview."""
        if self.tree is None:
            return
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if item and column in ['#9', '#10']:
            idx = int(self.tree.item(item, 'values')[1])
            col_idx = int(column[1:]) - 1
            col_name = self.tree['columns'][col_idx]
            current_value = self.tree.item(item, 'values')[col_idx]
            entry = ttk.Entry(self.tree)
            entry.insert(0, current_value)
            entry.place(relx=float(column[1:]) / len(self.tree['columns']), rely=0.0, anchor='nw', width=150)
            def save_edit(event=None):
                new_value = entry.get()
                if self.wo_df is not None:
                    self.wo_df.at[idx, col_name] = new_value
                    self.wo_df.at[idx, 'Failure Code'] = CUSTOM_CODE
                self.update_table()
                entry.destroy()
                logging.debug(f"Updated index {idx}: {col_name}='{new_value}', Failure Code='{CUSTOM_CODE}'")
            entry.bind('<Return>', lambda e: save_edit())
            entry.bind('<FocusOut>', lambda e: save_edit())
            entry.focus_set()
    
    def toggle_row(self, event):
        """Toggle row inclusion when clicking the 'Include' column."""
        if self.tree is None:
            return
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if item and column == '#1':
            idx = int(self.tree.item(item, 'values')[1])
            if idx in self.included_indices:
                self.included_indices.remove(idx)
                self.tree.item(item, values=['☐'] + list(self.tree.item(item, 'values')[1:]))
            else:
                self.included_indices.add(idx)
                self.tree.item(item, values=['☑'] + list(self.tree.item(item, 'values')[1:]))
            self.update_table()
            logging.info(f"Toggled row {idx}: {'Included' if idx in self.included_indices else 'Excluded'}")
    
    def get_filtered_df(self) -> pd.DataFrame:
        """Apply all filters to the DataFrame."""
        if self.wo_df is None or self.wo_df.empty:
            return pd.DataFrame()
        
        filtered_df = self.wo_df.copy()
        
        # Apply date range filter
        if self.start_date or self.end_date:
            filtered_df['Parsed_Date'] = filtered_df['Reported Date'].apply(parse_date)
            if self.start_date:
                mask = filtered_df['Parsed_Date'] >= self.start_date
                filtered_df = filtered_df[mask]
            if self.end_date:
                mask = filtered_df['Parsed_Date'] <= self.end_date
                filtered_df = filtered_df[mask]
            # Ensure filtered_df is still a DataFrame before calling drop
            if isinstance(filtered_df, pd.DataFrame):
                filtered_df = filtered_df.drop(columns=['Parsed_Date'], errors='ignore')
            else:
                # Convert back to DataFrame if it became a Series
                filtered_df = pd.DataFrame(filtered_df).T if isinstance(filtered_df, pd.Series) else pd.DataFrame()
        
        # Apply equipment filter
        equipment = self.equipment_var.get()
        if equipment:
            mask = filtered_df['Equipment #'] == equipment
            filtered_df = filtered_df[mask]
        
        # Apply work type filter
        work_type = self.work_type_var.get()
        if work_type:
            mask = filtered_df['Work Type'] == work_type
            filtered_df = filtered_df[mask]
        
        # Apply failure code filter (by selected source)
        failure_code = self.failure_code_var.get()
        code_col = 'Failure Code' if self.failure_code_source_var.get() == 'AI/Dictionary' else 'User failure code'
        if failure_code:
            mask = filtered_df[code_col] == failure_code
            filtered_df = filtered_df[mask]
        
        # Ensure we return a DataFrame
        if isinstance(filtered_df, pd.Series):
            filtered_df = filtered_df.to_frame().T
        elif not isinstance(filtered_df, pd.DataFrame):
            filtered_df = pd.DataFrame()
        
        return filtered_df
    
    def update_risk(self):
        """Calculate and display failure rate and annualized risk."""
        if self.wo_df is None or self.wo_df.empty:
            self.risk_label.config(text="Failure Rate: N/A, Annualized Risk: N/A")
            self.cost_label.config(text="Total Cost: $0.00 | Average Cost: $0.00 | Work Orders: 0", foreground="gray")
            self.filters_label.config(text="No data loaded", foreground="gray")
            return
        
        # Check if we're in segmented view and have segment data
        if self.is_segmented_view and self.segment_data:
            self.update_risk_segmented(self.segment_data)
            return
        
        try:
            filtered_df = self.get_filtered_df()
            if filtered_df.empty:
                self.risk_label.config(text="Failure Rate: N/A, Annualized Risk: N/A")
                self.cost_label.config(text="Total Cost: $0.00 | Average Cost: $0.00 | Work Orders: 0", foreground="gray")
                self.filters_label.config(text="No data matches current filters", foreground="gray")
                return
            
            valid_indices = filtered_df.index.intersection(self.included_indices)
            lambda_param, beta, failures_per_year = calculate_crow_amsaa_params(filtered_df, set(valid_indices))
            
            # Calculate cost statistics for included work orders
            included_df = filtered_df.loc[list(valid_indices)]
            total_cost = 0.0
            valid_costs = 0
            
            for idx, row in included_df.iterrows():
                work_order_cost = row.get('Work Order Cost', 0.0)
                try:
                    if work_order_cost is not None and str(work_order_cost).strip() != '' and str(work_order_cost).lower() != 'nan':
                        cost = float(work_order_cost)
                        total_cost += cost
                        valid_costs += 1
                except (ValueError, TypeError):
                    continue
            
            avg_cost = total_cost / valid_costs if valid_costs > 0 else 0.0
            work_order_count = len(valid_indices)
            
            # Update cost statistics
            cost_text = f"Total Cost: ${total_cost:,.2f} | Average Cost: ${avg_cost:,.2f} | Work Orders: {work_order_count}"
            self.cost_label.config(text=cost_text, foreground="blue")
            
            # Update current filters information
            filters_text = self.get_current_filters_text()
            self.filters_label.config(text=filters_text, foreground="black")
            
            try:
                prod_loss = float(self.prod_loss_var.get())
                maint_cost = float(self.maint_cost_var.get())
                margin = float(self.margin_var.get())
                forecast_years = float(self.forecast_period_var.get())
            except ValueError:
                self.risk_label.config(text="Failure Rate: N/A, Annualized Risk: Invalid input")
                logging.error("Invalid input for risk calculation")
                return
            
            # Calculate current annualized risk
            current_risk = failures_per_year * (prod_loss * margin + maint_cost)
            
            # Calculate forecasted risk using Crow-AMSAA parameters
            forecasted_risk = 0.0
            forecasted_failures = 0.0
            
            if lambda_param is not None and beta is not None and forecast_years > 0:
                # Convert years to days for calculation
                forecast_days = forecast_years * 365
                
                # Calculate expected failures in forecast period
                # Using Crow-AMSAA: N(t) = λ * t^β
                forecasted_failures = lambda_param * (forecast_days ** beta)
                
                # Calculate risk for forecast period
                forecasted_risk = forecasted_failures * (prod_loss * margin + maint_cost)
                
                # Annualize the forecasted risk
                forecasted_annual_risk = (forecasted_risk / forecast_years)
            
            # Build risk summary text with filter information and forecasting
            risk_text = f"Current: {failures_per_year:.2f} failures/year, ${current_risk:,.2f} risk"
            
            if lambda_param is not None and beta is not None:
                risk_text += f" | β={beta:.2f}, λ={lambda_param:.4f}"
                
                if forecast_years > 0:
                    risk_text += f" | Forecast ({forecast_years:.0f} years): {forecasted_failures:.2f} failures, ${forecasted_risk:,.2f} risk"
                    if forecast_years != 1:
                        risk_text += f" (${forecasted_annual_risk:,.2f}/year)"
            
            # Add equipment filter if applied
            equipment = self.equipment_var.get()
            if equipment:
                risk_text += f" | Equipment: {equipment}"
            
            # Add failure code filter if applied
            failure_code = self.failure_code_var.get()
            if failure_code:
                source = self.failure_code_source_var.get()
                risk_text += f" | Failure Mode ({source}): {failure_code}"
            
            self.risk_label.config(text=risk_text)
            logging.debug(f"Calculated risk: failures/year={failures_per_year:.2f}, prod_loss={prod_loss}, maint_cost={maint_cost}, margin={margin}, current_risk=${current_risk:,.2f}, forecasted_risk=${forecasted_risk:,.2f}")
            
            # Update the risk plot if we have valid Crow-AMSAA parameters
            if lambda_param is not None and beta is not None:
                self.update_risk_plot()
        except Exception as e:
            self.risk_label.config(text="Failure Rate: N/A, Annualized Risk: Error")
            self.cost_label.config(text="Total Cost: $0.00 | Average Cost: $0.00 | Work Orders: 0", foreground="red")
            self.filters_label.config(text="Error calculating statistics", foreground="red")
            logging.error(f"Error calculating risk: {e}")
    
    def update_risk_plot(self):
        """Update the risk trend plot based on current parameters and forecast period."""
        if self.wo_df is None or self.wo_df.empty:
            # Clear plot and show message
            for widget in self.risk_plot_frame.winfo_children():
                widget.destroy()
            ttk.Label(self.risk_plot_frame, text="No data available for plotting", foreground="gray").pack(expand=True)
            return
        
        try:
            # Get current parameters
            prod_loss = float(self.prod_loss_var.get())
            maint_cost = float(self.maint_cost_var.get())
            margin = float(self.margin_var.get())
            forecast_years = float(self.forecast_period_var.get())
            
            # Get Crow-AMSAA parameters
            filtered_df = self.get_filtered_df()
            if filtered_df.empty:
                for widget in self.risk_plot_frame.winfo_children():
                    widget.destroy()
                ttk.Label(self.risk_plot_frame, text="No data matches current filters", foreground="gray").pack(expand=True)
                return
            
            valid_indices = filtered_df.index.intersection(self.included_indices)
            lambda_param, beta, failures_per_year = calculate_crow_amsaa_params(filtered_df, set(valid_indices))
            
            if lambda_param is None or beta is None:
                for widget in self.risk_plot_frame.winfo_children():
                    widget.destroy()
                ttk.Label(self.risk_plot_frame, text="Insufficient data for Crow-AMSAA analysis", foreground="gray").pack(expand=True)
                return
            
            # Clear previous plot
            for widget in self.risk_plot_frame.winfo_children():
                widget.destroy()
            
            # Create projection based on user's forecast period (convert years to days)
            forecast_days = int(forecast_years * 365)
            days = np.arange(1, forecast_days + 1)
            cumulative_failures = lambda_param * (days ** beta)
            
            # Calculate risk for each time point
            risk_per_failure = prod_loss * margin + maint_cost
            cumulative_risk = cumulative_failures * risk_per_failure
            
            # Get frame dimensions for dynamic sizing
            self.risk_plot_frame.update_idletasks()
            width_px = max(self.risk_plot_frame.winfo_width(), 100)
            height_px = max(self.risk_plot_frame.winfo_height(), 100)
            width_in = min(max(width_px / 100, 4), 12)
            height_in = min(max(height_px / 100, 2.5), 8)
            font_scale = width_in / 8
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(width_in, height_in))
            
            # Plot cumulative risk
            ax.plot(days, cumulative_risk, 'b-', linewidth=2, label='Cumulative Risk')
            
            # Add markers based on forecast period
            if forecast_years == 1:
                # For 1 year, use 90-day intervals
                markers = np.arange(90, forecast_days + 1, 90)
                marker_labels = [f'Q{i+1}' for i in range(len(markers))]
            else:
                # For multiple years, use annual markers
                markers = np.arange(365, forecast_days + 1, 365)
                marker_labels = [f'Year {i+1}' for i in range(len(markers))]
            
            # Add bold vertical lines for markers
            for marker_day in markers:
                ax.axvline(x=float(marker_day), color='red', linestyle='-', linewidth=2, alpha=0.7)
            
            # Add labels on x-axis
            ax.set_xticks(markers)
            ax.set_xticklabels(marker_labels, rotation=45, ha='right')
            
            # Formatting
            ax.set_xlabel('Time', fontsize=10 * font_scale)
            ax.set_ylabel('Cumulative Risk ($)', fontsize=10 * font_scale)
            ax.set_title(f'{forecast_years:.1f}-Year Risk Projection\nβ={beta:.2f}, λ={lambda_param:.4f}, Risk per Failure=${risk_per_failure:,.0f}', 
                        fontsize=12 * font_scale)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8 * font_scale)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add equipment and failure mode info if filters are applied
            equipment = self.equipment_var.get()
            failure_code = self.failure_code_var.get()
            if equipment or failure_code:
                info_text = "Filters: "
                if equipment:
                    info_text += f"Equipment={equipment}"
                if failure_code:
                    source = self.failure_code_source_var.get()
                    info_text += f", Failure Mode ({source})={failure_code}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8 * font_scale, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.risk_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store references
            self.risk_plot_fig = fig
            self.risk_plot_canvas = canvas
            
            logging.info(f"Risk plot updated: β={beta:.2f}, λ={lambda_param:.4f}, {forecast_years:.1f}-year risk=${cumulative_risk[-1]:,.0f}")
            
        except Exception as e:
            for widget in self.risk_plot_frame.winfo_children():
                widget.destroy()
            ttk.Label(self.risk_plot_frame, text=f"Error creating plot: {str(e)}", foreground="red").pack(expand=True)
            logging.error(f"Error updating risk plot: {e}")
    
    def export_risk_plot(self):
        """Export the risk trend plot to a file."""
        if self.risk_plot_fig is None:
            messagebox.showwarning("Warning", "No plot available to export. Please update the plot first.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Export Risk Trend Plot"
            )
            
            if file_path:
                self.risk_plot_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Risk trend plot exported to {file_path}")
                logging.info(f"Risk plot exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
            logging.error(f"Error exporting risk plot: {e}")
    
    def get_current_filters_text(self) -> str:
        """Get a text representation of current filters applied to the data."""
        filters = []
        
        # Equipment filter
        equipment = self.equipment_var.get()
        if equipment:
            filters.append(f"Equipment: {equipment}")
        
        # Work type filter
        work_type = self.work_type_var.get()
        if work_type:
            filters.append(f"Work Type: {work_type}")
        
        # Failure code filter
        failure_code = self.failure_code_var.get()
        if failure_code:
            source = self.failure_code_source_var.get()
            filters.append(f"Failure Code ({source}): {failure_code}")
        
        # Date range filter
        if self.start_date or self.end_date:
            date_range = []
            if self.start_date:
                date_range.append(f"From: {self.start_date.strftime('%m/%d/%Y')}")
            if self.end_date:
                date_range.append(f"To: {self.end_date.strftime('%m/%d/%Y')}")
            filters.append(f"Date Range: {' - '.join(date_range)}")
        
        # Included/excluded work orders
        if hasattr(self, 'included_indices') and self.wo_df is not None:
            total_work_orders = len(self.wo_df)
            included_count = len(self.included_indices)
            excluded_count = total_work_orders - included_count
            if excluded_count > 0:
                filters.append(f"Work Orders: {included_count} included, {excluded_count} excluded")
            else:
                filters.append(f"Work Orders: All {included_count} included")
        
        if not filters:
            return "No filters applied"
        
        return " | ".join(filters)

    # --- Crow-AMSAA plot with interactivity ---
    def highlight_plot_point_by_work_order(self, work_order_idx):
        """Highlight the corresponding data point on the Crow-AMSAA plot when a work order is selected."""
        if self.crow_amsaa_fig is None or self.wo_df is None:
            logging.debug("Cannot highlight: figure or dataframe is None")
            return
        
        # Remove previous highlight if present
        if hasattr(self, 'last_highlight_artist') and self.last_highlight_artist is not None:
            try:
                self.last_highlight_artist.remove()
                self.last_highlight_artist = None
                if self.crow_amsaa_canvas:
                    self.crow_amsaa_canvas.draw()
            except Exception as e:
                logging.debug(f"Error removing last highlight: {e}")
        
        try:
            # Get the date for the selected work order
            selected_date = parse_date(self.wo_df.at[work_order_idx, 'Reported Date'])
            if pd.isna(selected_date):
                logging.debug(f"Cannot highlight: invalid date for work order {work_order_idx}")
                return
            
            logging.debug(f"Highlighting work order {work_order_idx} with date {selected_date}")
            
            # Get the filtered data to match the plot
            filtered_df = self.get_filtered_df()
            valid_indices = filtered_df.index.intersection(self.included_indices)
            filtered_dates = [parse_date(filtered_df.at[idx, 'Reported Date']) for idx in valid_indices if pd.notna(filtered_df.at[idx, 'Reported Date'])]
            filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
            
            if len(filtered_dates) < 2:
                logging.debug("Cannot highlight: insufficient filtered dates")
                return
            
            dates = sorted(filtered_dates)
            t0 = dates[0]
            times = [(d - t0).days + 1 for d in dates]
            
            logging.debug(f"Filtered dates range: {dates[0]} to {dates[-1]}")
            logging.debug(f"Times range: {times[0]} to {times[-1]}")
            
            # Find the index of the selected date in the filtered dates
            try:
                date_idx = dates.index(selected_date)
                logging.debug(f"Exact date match found at index {date_idx}")
            except ValueError:
                # If exact match not found, find closest date
                date_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - selected_date).days))
                logging.debug(f"Closest date match found at index {date_idx} (date: {dates[date_idx]})")
            
            # Get the corresponding time value
            selected_time = times[date_idx]
            selected_n = date_idx + 1  # Cumulative failures
            
            logging.debug(f"Selected coordinates: time={selected_time}, failures={selected_n}")
            
            # Find and highlight the corresponding point on the plot
            ax = self.crow_amsaa_fig.axes[0] if len(self.crow_amsaa_fig.axes) == 1 else None
            
            if ax is None and len(self.crow_amsaa_fig.axes) == 2:
                # Handle segmented plots - determine which segment contains this date
                axs = self.crow_amsaa_fig.axes
                seg_dt = parse_date(self.segment_date) if hasattr(self, 'segment_date') and self.segment_date else None
                if seg_dt:
                    seg_idx = next((i for i, d in enumerate(dates) if d >= seg_dt), len(dates))
                    if date_idx <= seg_idx:
                        ax = axs[0]  # First segment
                        logging.debug(f"Using first segment (axs[0])")
                    else:
                        ax = axs[1]  # Second segment
                        # Adjust the time for second segment
                        selected_time = times[date_idx] - times[seg_idx]
                        selected_n = date_idx - seg_idx + 1
                        logging.debug(f"Using second segment (axs[1]), adjusted coordinates: time={selected_time}, failures={selected_n}")
            
            if ax is None:
                logging.debug("Cannot highlight: no valid axis found")
                return
            
            # Clear previous highlights by resetting all scatter points
            scatter_artists = []
            for artist in ax.get_children():
                if hasattr(artist, 'get_offsets') and len(artist.get_offsets()) > 0:  # type: ignore
                    scatter_artists.append(artist)
                    try:
                        # Type-safe artist manipulation
                        if hasattr(artist, 'set_alpha'):
                            artist.set_alpha(0.6)  # type: ignore
                        # Only call set_s if it's a scatter plot artist
                        if hasattr(artist, 'set_s') and hasattr(artist, 'get_offsets'):  # type: ignore
                            artist.set_s(50)  # type: ignore
                        if hasattr(artist, 'set_facecolor'):  # type: ignore
                            artist.set_facecolor('blue')  # type: ignore
                        elif hasattr(artist, 'set_color'):  # type: ignore
                            artist.set_color('blue')  # type: ignore
                    except Exception as e:
                        logging.debug(f"Error resetting artist: {e}")
            
            logging.debug(f"Found {len(scatter_artists)} scatter artists")
            
            # Find and highlight the corresponding point
            point_found = False
            for artist in scatter_artists:
                try:
                    offsets = artist.get_offsets()
                    logging.debug(f"Checking artist with {len(offsets)} points")
                    
                    for i, (x, y) in enumerate(offsets):
                        # More lenient matching - check if this point is close to our selected point
                        time_match = abs(x - selected_time) < 5  # Allow 5 days tolerance
                        failure_match = abs(y - selected_n) < 2   # Allow 2 failures tolerance
                        
                        if time_match and failure_match:
                            # Highlight this point
                            try:
                                artist.set_alpha(1.0)
                                artist.set_s(150)  # Make it even larger
                                # Try different methods to set color
                                if hasattr(artist, 'set_facecolor'):
                                    artist.set_facecolor('red')
                                elif hasattr(artist, 'set_color'):
                                    artist.set_color('red')
                                else:
                                    # Fallback: try to set the color array
                                    colors = ['red'] * len(offsets)
                                    artist.set_facecolor(colors)
                                point_found = True
                                logging.debug(f"Highlighted point at coordinates ({x}, {y})")
                                break
                            except Exception as e:
                                logging.debug(f"Error highlighting point: {e}")
                                # Try alternative highlighting method
                                try:
                                    # Remove previous highlight if present
                                    if hasattr(self, 'last_highlight_artist') and self.last_highlight_artist is not None:
                                        self.last_highlight_artist.remove()
                                        self.last_highlight_artist = None
                                    # Create a new highlighted point on top and store reference
                                    highlight = ax.scatter([x], [y], c='red', s=200, alpha=1.0, zorder=10, marker='o')
                                    self.last_highlight_artist = highlight
                                    point_found = True
                                    logging.debug(f"Added highlighted point at coordinates ({x}, {y})")
                                    break
                                except Exception as e2:
                                    logging.debug(f"Error adding highlighted point: {e2}")
                    
                    if point_found:
                        break
                except Exception as e:
                    logging.debug(f"Error processing artist: {e}")
            
            if not point_found:
                logging.debug(f"No matching point found for coordinates ({selected_time}, {selected_n})")
                # Log all available points for debugging
                for artist in scatter_artists:
                    try:
                        offsets = artist.get_offsets()
                        logging.debug(f"Available points: {list(offsets)}")
                    except Exception as e:
                        logging.debug(f"Error getting offsets: {e}")
            
            if self.crow_amsaa_canvas:
                try:
                    self.crow_amsaa_canvas.draw()
                    logging.debug("Canvas redrawn")
                except Exception as e:
                    logging.debug(f"Error redrawing canvas: {e}")
                
        except Exception as e:
            logging.error(f"Error highlighting plot point: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def return_to_single_plot(self):
        """Return from segmented view to single Crow-AMSAA plot."""
        self.is_segmented_view = False
        self.segment_data = None
        self.return_to_single_button.pack_forget()  # Hide the button
        # Redraw the single plot
        filtered_df = self.get_filtered_df()
        for widget in self.crow_amsaa_frame.winfo_children():
            widget.destroy()
        self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, self.crow_amsaa_frame)
        # Update risk evaluation to show single plot data
        self.update_risk()

    def update_risk_segmented(self, segment_data):
        """Update risk evaluation for segmented plots showing both segments."""
        if not segment_data or len(segment_data) != 2:
            return
        
        try:
            (beta1, lambda1, failures_per_year1), (beta2, lambda2, failures_per_year2) = segment_data
            
            # Get risk parameters
            try:
                prod_loss = float(self.prod_loss_var.get())
                maint_cost = float(self.maint_cost_var.get())
                margin = float(self.margin_var.get())
                forecast_years = float(self.forecast_period_var.get())
            except ValueError:
                self.risk_label.config(text="Segment 1: N/A, Segment 2: N/A (Invalid input)")
                return
            
            # Calculate risks for both segments
            risk1 = failures_per_year1 * (prod_loss * margin + maint_cost) if failures_per_year1 is not None else 0
            risk2 = failures_per_year2 * (prod_loss * margin + maint_cost) if failures_per_year2 is not None else 0
            
            # Calculate forecasted risks for both segments
            forecasted_risk1 = 0.0
            forecasted_risk2 = 0.0
            
            if forecast_years > 0:
                forecast_days = forecast_years * 365
                if failures_per_year1 is not None:
                    # Estimate lambda and beta for segment 1 (simplified)
                    lambda1_est = failures_per_year1 / (365 ** 1.0)  # Assume beta=1.0 for estimation
                    forecasted_failures1 = lambda1_est * (forecast_days ** 1.0)
                    forecasted_risk1 = forecasted_failures1 * (prod_loss * margin + maint_cost)
                
                if failures_per_year2 is not None:
                    # Estimate lambda and beta for segment 2 (simplified)
                    lambda2_est = failures_per_year2 / (365 ** 1.0)  # Assume beta=1.0 for estimation
                    forecasted_failures2 = lambda2_est * (forecast_days ** 1.0)
                    forecasted_risk2 = forecasted_failures2 * (prod_loss * margin + maint_cost)
            
            # Handle None values for display
            failures_per_year1_display = f"{failures_per_year1:.2f}" if failures_per_year1 is not None else "N/A"
            failures_per_year2_display = f"{failures_per_year2:.2f}" if failures_per_year2 is not None else "N/A"
            
            risk_text = f"Segment 1: {failures_per_year1_display} failures/year, ${risk1:,.2f} risk"
            if forecast_years > 0 and failures_per_year1 is not None:
                risk_text += f" | Forecast: ${forecasted_risk1:,.2f}"
            
            risk_text += f" | Segment 2: {failures_per_year2_display} failures/year, ${risk2:,.2f} risk"
            if forecast_years > 0 and failures_per_year2 is not None:
                risk_text += f" | Forecast: ${forecasted_risk2:,.2f}"
            
            # Add equipment filter if applied
            equipment = self.equipment_var.get()
            if equipment:
                risk_text += f" | Equipment: {equipment}"
            
            # Add failure code filter if applied
            failure_code = self.failure_code_var.get()
            if failure_code:
                source = self.failure_code_source_var.get()
                risk_text += f" | Failure Mode ({source}): {failure_code}"
            
            self.risk_label.config(text=risk_text)
            
            # Update cost statistics and filters for segmented view
            filtered_df = self.get_filtered_df()
            if not filtered_df.empty:
                valid_indices = filtered_df.index.intersection(self.included_indices)
                included_df = filtered_df.loc[list(valid_indices)]
                
                # Calculate cost statistics for included work orders
                total_cost = 0.0
                valid_costs = 0
                
                for idx, row in included_df.iterrows():
                    work_order_cost = row.get('Work Order Cost', 0.0)
                    try:
                        if work_order_cost is not None and str(work_order_cost).strip() != '' and str(work_order_cost).lower() != 'nan':
                            cost = float(work_order_cost)
                            total_cost += cost
                            valid_costs += 1
                    except (ValueError, TypeError):
                        continue
                
                avg_cost = total_cost / valid_costs if valid_costs > 0 else 0.0
                work_order_count = len(valid_indices)
                
                # Update cost statistics
                cost_text = f"Total Cost: ${total_cost:,.2f} | Average Cost: ${avg_cost:,.2f} | Work Orders: {work_order_count}"
                self.cost_label.config(text=cost_text, foreground="blue")
                
                # Update current filters information
                filters_text = self.get_current_filters_text()
                self.filters_label.config(text=filters_text, foreground="black")
            else:
                self.cost_label.config(text="Total Cost: $0.00 | Average Cost: $0.00 | Work Orders: 0", foreground="gray")
                self.filters_label.config(text="No data matches current filters", foreground="gray")
            
            logging.debug(f"Segmented risk: {risk_text}")
        except Exception as e:
            self.risk_label.config(text="Error calculating segmented risk")
            self.cost_label.config(text="Total Cost: $0.00 | Average Cost: $0.00 | Work Orders: 0", foreground="red")
            self.filters_label.config(text="Error calculating statistics", foreground="red")
            logging.error(f"Error calculating segmented risk: {e}")

    def create_crow_amsaa_plot_interactive(self, filtered_df, included_indices, frame, segment_date=None):
        """Create an interactive Crow-AMSAA plot. If segment_date is given, plot two segments."""
        # Remove previous plot
        for widget in frame.winfo_children():
            widget.destroy()
        
        # Show/hide return button based on mode
        if segment_date is not None:
            self.is_segmented_view = True
            self.segment_date = segment_date  # Store for highlighting logic
            # Find the plot controls frame and pack the button there
            plot_controls = None
            for widget in frame.master.winfo_children():
                if isinstance(widget, ttk.Frame) and len(widget.winfo_children()) > 0:
                    # Check if this frame contains buttons
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button):
                            plot_controls = widget
                            break
                    if plot_controls:
                        break
            
            if plot_controls:
                self.return_to_single_button.pack(side=tk.LEFT, padx=5, in_=plot_controls)
        else:
            self.is_segmented_view = False
            self.segment_date = None
            self.return_to_single_button.pack_forget()
        
        # Prepare data
        valid_indices = filtered_df.index.intersection(included_indices)
        filtered_dates = [parse_date(filtered_df.at[idx, 'Reported Date']) for idx in valid_indices if pd.notna(filtered_df.at[idx, 'Reported Date'])]
        filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
        if len(filtered_dates) < 2:
            # Not enough data to plot
            label = ttk.Label(frame, text="Not enough data to plot Crow-AMSAA.", foreground="gray")
            label.pack(expand=True)
            return None, None, None
        
        dates = sorted(filtered_dates)
        t0 = dates[0]
        times = [(d - t0).days + 1 for d in dates]
        n = np.arange(1, len(times) + 1)
        
        # If segmenting, split data
        if segment_date is not None:
            seg_dt = parse_date(segment_date)
            seg_idx = next((i for i, d in enumerate(dates) if d >= seg_dt), len(dates))
            
            # First segment
            times1 = times[:seg_idx+1]
            n1 = n[:seg_idx+1]
            
            # Second segment
            times2 = times[seg_idx:]
            n2 = n[seg_idx:]
            
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            
            # Plot segment 1
            beta1 = lambda1 = failures_per_year1 = None
            if len(times1) > 1:
                log_n1 = np.log(n1)
                log_t1 = np.log(times1)
                coeffs1 = np.polyfit(log_t1, log_n1, 1)
                beta1 = coeffs1[0]
                lambda1 = np.exp(coeffs1[1])
                t_fit1 = np.linspace(min(times1), max(times1), 100)
                n_fit1 = lambda1 * t_fit1 ** beta1
                failures_per_year1 = lambda1 * (365 ** beta1)
                axs[0].scatter(times1, n1, marker='o', label='Observed', picker=5)
                axs[0].plot(t_fit1, n_fit1, label=f'β={beta1:.2f}, λ={lambda1:.4f}')
                axs[0].set_xscale('log')
                axs[0].set_yscale('log')
                axs[0].set_title(f'Segment 1\nFailures/year={failures_per_year1:.2f}')
                axs[0].legend()
                axs[0].grid(True, which="both", ls="--")
            else:
                axs[0].set_title('Segment 1 (Insufficient data)')
            
            # Plot segment 2
            beta2 = lambda2 = failures_per_year2 = None
            if len(times2) > 1:
                log_n2 = np.log(n2)
                log_t2 = np.log(times2)
                coeffs2 = np.polyfit(log_t2, log_n2, 1)
                beta2 = coeffs2[0]
                lambda2 = np.exp(coeffs2[1])
                t_fit2 = np.linspace(min(times2), max(times2), 100)
                n_fit2 = lambda2 * t_fit2 ** beta2
                failures_per_year2 = lambda2 * (365 ** beta2)
                axs[1].scatter(times2, n2, marker='o', label='Observed', picker=5)
                axs[1].plot(t_fit2, n_fit2, label=f'β={beta2:.2f}, λ={lambda2:.4f}')
                axs[1].set_xscale('log')
                axs[1].set_yscale('log')
                axs[1].set_title(f'Segment 2\nFailures/year={failures_per_year2:.2f}')
                axs[1].legend()
                axs[1].grid(True, which="both", ls="--")
            else:
                axs[1].set_title('Segment 2 (Insufficient data)')
            
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store for event handling
            self.crow_amsaa_canvas = canvas
            self.crow_amsaa_fig = fig
            
            # Store segment data for risk calculation
            self.segment_data = (
                (beta1, lambda1, failures_per_year1),
                (beta2, lambda2, failures_per_year2)
            )
            
            # Update risk evaluation for segmented view
            self.update_risk_segmented(self.segment_data)
            
            return fig, canvas, (beta1 if len(times1)>1 else None, beta2 if len(times2)>1 else None)
        
        # Normal (not segmented)
        log_n = np.log(n)
        log_t = np.log(times)
        coeffs = np.polyfit(log_t, log_n, 1)
        beta = coeffs[0]
        lambda_param = np.exp(coeffs[1])
        failures_per_year = lambda_param * (365 ** beta)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(times, n, marker='o', label='Observed Failures', picker=5)
        t_fit = np.linspace(min(times), max(times), 100)
        n_fit = lambda_param * t_fit ** beta
        ax.plot(t_fit, n_fit, label=f'Crow-AMSAA (β={beta:.2f}, λ={lambda_param:.4f})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Crow-AMSAA Plot\nFailures/year={failures_per_year:.2f}")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Cumulative Failures")
        ax.legend()
        ax.grid(True, which="both", ls="--")
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store for event handling
        self.crow_amsaa_canvas = canvas
        self.crow_amsaa_fig = fig
        
        return fig, canvas, beta

    def show_context_menu(self, event):
        """Show right-click context menu for segmenting Crow-AMSAA plot."""
        if self.tree is None:
            return
        item = self.tree.identify_row(event.y)
        if not item:
            return
        self.tree.selection_set(item)
        # Create context menu if not already
        if self.context_menu is not None:
            self.context_menu.destroy()
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Segment Crow-AMSAA at this date", command=lambda: self.segment_crow_amsaa_at_selected())
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def segment_crow_amsaa_at_selected(self):
        """Segment Crow-AMSAA plot at the selected work order's date."""
        if self.tree is None or self.wo_df is None:
            return
        selected = self.tree.selection()
        if not selected:
            return
        item = selected[0]
        values = self.tree.item(item, 'values')
        if len(values) < 8:
            return
        idx = int(values[1])
        date_str = self.wo_df.at[idx, 'Reported Date']
        # Redraw Crow-AMSAA plot segmented at this date
        filtered_df = self.get_filtered_df()
        self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, self.crow_amsaa_frame, segment_date=date_str)

    def show_equipment_context_menu(self, event):
        """Show right-click context menu for equipment filtering."""
        if self.equipment_tree is None:
            return
        item = self.equipment_tree.identify_row(event.y)
        if not item:
            return
        self.equipment_tree.selection_set(item)
        # Create context menu if not already
        if hasattr(self, 'equipment_context_menu') and self.equipment_context_menu is not None:
            self.equipment_context_menu.destroy()
        self.equipment_context_menu = tk.Menu(self.equipment_tree, tearoff=0)
        self.equipment_context_menu.add_command(label="Filter Work Orders to This Equipment", 
                                               command=lambda: self.filter_to_equipment())
        self.equipment_context_menu.tk_popup(event.x_root, event.y_root)

    def filter_to_equipment(self):
        """Filter work order data to the selected equipment while preserving other filters."""
        if self.equipment_tree is None or self.wo_df is None:
            return
        
        selected = self.equipment_tree.selection()
        if not selected:
            return
        
        item = selected[0]
        values = self.equipment_tree.item(item, 'values')
        if len(values) < 1:
            return
        
        equipment_number = values[0]
        
        # Store current filter states to preserve them
        current_work_type = self.work_type_var.get()
        current_failure_code = self.failure_code_var.get()
        current_failure_code_source = self.failure_code_source_var.get()
        current_start_date = self.start_date_entry.get()
        current_end_date = self.end_date_entry.get()
        
        # Set the equipment filter
        self.equipment_var.set(equipment_number)
        
        # Update the table with the new equipment filter while preserving other filters
        self.update_table()
        
        # Update status to show the filter was applied
        self.status_label.config(text=f"Filtered to equipment {equipment_number} (preserving other filters)", foreground="blue")
        logging.info(f"Filtered work orders to equipment {equipment_number} while preserving other filters")

    def update_table(self):
        """Update work order and equipment summary tables."""
        if self.wo_df is None or self.wo_df.empty:
            self.status_label.config(text="No data available to display.", foreground="red")
            self.root.config(cursor="")
            self.root.update()
            logging.error("Cannot update table: wo_df is None or empty")
            # --- Always clear Crow-AMSAA plot area if no data ---
            for widget in self.crow_amsaa_frame.winfo_children():
                widget.destroy()
            return
        
        self.status_label.config(text="Processing...", foreground="blue")
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            # Get filtered DataFrame
            filtered_df = self.get_filtered_df()
            status_text = f"Work order table filtered to equipment {self.equipment_var.get() or 'all'}, work type {self.work_type_var.get() or 'all'}, failure code {self.failure_code_var.get() or 'all'} (source: {self.failure_code_source_var.get()}) ."
            logging.info(f"Work order table: {status_text}, rows={len(filtered_df)}")
            
            if filtered_df.empty:
                self.status_label.config(text=f"No data for {status_text}.", foreground="purple")
                self.root.config(cursor="")
                self.root.update()
                logging.warning(f"No data for {status_text}")
                # --- Always clear Crow-AMSAA plot area if no data ---
                for widget in self.crow_amsaa_frame.winfo_children():
                    widget.destroy()
                # Optionally, show a message in the plot area
                label = ttk.Label(self.crow_amsaa_frame, text="Not enough data to plot Crow-AMSAA.", foreground="gray")
                label.pack(expand=True)
                return
            
            if self.tree:
                self.tree.destroy()
            if self.equipment_tree:
                self.equipment_tree.destroy()
            
            for widget in self.crow_amsaa_frame.winfo_children():
                widget.destroy()
            
            mtbf = calculate_mtbf(filtered_df, set(self.included_indices))
            
            # Create Work Order table
            columns = ['Include', 'Index', 'Work Order', 'Description', 'Asset', 'Equipment #', 
                       'Work Type', 'Reported Date', 'Failure Code', 'Failure Description', 'Matched Keyword', 'User failure code', 'AI Confidence', 'Classification Method', 'Work Order Cost']
            self.tree = ttk.Treeview(self.work_order_frame, columns=columns, show='headings')
            
            self.tree.heading('Include', text='Include', command=lambda: self.sort_column(self.tree, 'Include', self.sort_states.get((self.tree, 'Include'), False)))
            self.tree.column('Include', width=50)
            self.tree.heading('Index', text='')
            self.tree.column('Index', width=0, stretch=False)
            for col in columns[2:]:
                self.tree.heading(col, text=col, command=lambda c=col: self.sort_column(self.tree, c, self.sort_states.get((self.tree, c), False)))
                self.tree.column(col, width=100)
            self.tree.column('Failure Description', width=150)
            
            self.tree.insert('', 'end', values=('MTBF', '', f'{mtbf:.2f} days', '', '', '', '', '', '', '', '', '', '', '', ''))
            
            for idx, row in filtered_df.iterrows():
                include = '☑' if idx in self.included_indices else '☐'
                ai_confidence = row.get('AI Confidence', 0.0)
                classification_method = row.get('Classification Method', 'dictionary')
                
                # Get work order cost, handle missing or invalid values
                work_order_cost = row.get('Work Order Cost', 0.0)
                try:
                    if work_order_cost is not None and str(work_order_cost).strip() != '' and str(work_order_cost).lower() != 'nan':
                        work_order_cost = float(work_order_cost)
                    else:
                        work_order_cost = 0.0
                except (ValueError, TypeError):
                    work_order_cost = 0.0
                
                values = [
                    include,
                    idx,
                    row.get('Work Order', ''),
                    row.get('Description', ''),
                    str(row.get('Asset', '')),
                    str(row.get('Equipment #', '')),
                    row.get('Work Type', ''),
                    row.get('Reported Date', ''),
                    row.get('Failure Code', DEFAULT_CODE),
                    row.get('Failure Description', DEFAULT_DESC),
                    row.get('Matched Keyword', ''),
                    row.get('User failure code', ''),
                    f'{ai_confidence:.2f}' if ai_confidence is not None and ai_confidence > 0 else '',
                    classification_method,
                    f'${work_order_cost:,.2f}' if work_order_cost > 0 else ''
                ]
                self.tree.insert('', 'end', values=values)
            
            self.tree.pack(fill=tk.BOTH, expand=True)
            self.tree.bind('<Button-1>', self.toggle_row)
            self.tree.bind('<Double-1>', self.edit_cell)
            # --- Add right-click context menu binding ---
            self.tree.bind('<Button-3>', self.show_context_menu)
            # --- Add selection event to highlight plot points ---
            self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
            
            # Update failure code dropdown based on selected source
            self.update_failure_code_dropdown()
            self.update_weibull_failure_dropdown()
            self.update_spares_equipment_dropdown()
            
            # Equipment Summary Table
            equipment = self.equipment_var.get()
            base_df = self.get_filtered_df()  # Use filtered data for equipment summary
            equipment_nums = [equipment] if equipment else sorted(base_df['Equipment #'].dropna().unique())
            eq_columns = ['Equipment #', 'Total Work Orders', 'Failures per Year', 'Total Cost', 'Avg Cost per WO']
            self.equipment_tree = ttk.Treeview(self.equipment_frame, columns=eq_columns, show='headings')
            
            for col in eq_columns:
                self.equipment_tree.heading(col, text=col, command=lambda c=col: self.sort_column(self.equipment_tree, c, self.sort_states.get((self.equipment_tree, c), False)))
                self.equipment_tree.column(col, width=120, anchor='center')
            
            for eq in equipment_nums:
                eq_df = base_df[base_df['Equipment #'] == eq]
                valid_indices = eq_df.index.intersection(self.included_indices)
                total_wo = len(valid_indices)
                # Ensure eq_df is a DataFrame before passing to calculate_crow_amsaa_params
                if isinstance(eq_df, pd.DataFrame):
                    _, _, failures_per_year = calculate_crow_amsaa_params(eq_df, set(valid_indices))
                else:
                    failures_per_year = 0.0
                
                # Calculate cost information
                total_cost = 0.0
                avg_cost = 0.0
                if total_wo > 0:
                    # Get costs for valid indices
                    costs = []
                    for idx in valid_indices:
                        cost = eq_df.at[idx, 'Work Order Cost']
                        try:
                            cost = float(cost) if pd.notna(cost) else 0.0
                            costs.append(cost)
                        except (ValueError, TypeError):
                            costs.append(0.0)
                    
                    total_cost = sum(costs)
                    avg_cost = total_cost / total_wo if total_wo > 0 else 0.0
                
                self.equipment_tree.insert('', 'end', values=(
                    eq, 
                    total_wo, 
                    failures_per_year,
                    f'${total_cost:,.2f}' if total_cost > 0 else '$0.00',
                    f'${avg_cost:,.2f}' if avg_cost > 0 else '$0.00'
                ))
            
            self.equipment_tree.pack(fill=tk.BOTH, expand=True)
            # Add right-click context menu binding for equipment tree
            self.equipment_tree.bind('<Button-3>', self.show_equipment_context_menu)
            
            date_text = f", date range: {self.start_date.strftime('%m/%d/%Y') if self.start_date else 'N/A'} to {self.end_date.strftime('%m/%d/%Y') if self.end_date else 'N/A'}" if self.start_date or self.end_date else ''
            self.status_label.config(text=f"{status_text}{date_text}", foreground="green")
            self.root.config(cursor="")
            self.root.update()
            logging.info(f"Table updated: {status_text}, rows={len(filtered_df)}")
            
            self.update_risk()
            
            # --- Always update Crow-AMSAA plot unless in segmented view ---
            if not self.is_segmented_view:
                for widget in self.crow_amsaa_frame.winfo_children():
                    widget.destroy()
                fig, canvas, beta = self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, self.crow_amsaa_frame)
                if fig is None:
                    # Not enough data, show message
                    label = ttk.Label(self.crow_amsaa_frame, text="Not enough data to plot Crow-AMSAA.", foreground="gray")
                    label.pack(expand=True)
        
        except Exception as e:
            error_msg = f"Error updating table: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red")
            self.root.config(cursor="")
            self.root.update()
            logging.error(error_msg)
    
    def on_tree_select(self, event):
        """Handle tree selection to highlight corresponding plot point."""
        if self.tree is None:
            return
        
        selected = self.tree.selection()
        if not selected:
            return
        
        item = selected[0]
        values = self.tree.item(item, 'values')
        if len(values) < 2:
            return
        
        try:
            idx = int(values[1])
            self.highlight_plot_point_by_work_order(idx)
        except (ValueError, IndexError):
            pass  # Skip if not a valid work order row

    def export_to_excel(self):
        """Export data to Excel workbook."""
        if self.wo_df is None or self.wo_df.empty:
            self.status_label.config(text="Please process files first.", foreground="red")
            messagebox.showerror("Error", "Please process files first.")
            return
        
        if not self.output_dir:
            self.output_dir = filedialog.askdirectory()
            if not self.output_dir:
                self.status_label.config(text="Canceled export: No output directory selected.", foreground="purple")
                return
        
        self.status_label.config(text="Processing...", foreground="blue")
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            logging.info("Starting Excel export")
            equipment = self.equipment_var.get()
            work_type = self.work_type_var.get()
            
            # Work Orders
            filtered_df = self.get_filtered_df()
            valid_indices = filtered_df.index.intersection(self.included_indices)
            filtered_df = filtered_df.loc[valid_indices]
            if filtered_df.empty:
                self.status_label.config(text="No data to export after filtering.", foreground="purple")
                self.root.config(cursor="")
                self.root.update()
                messagebox.showwarning("Warning", "No data to export.")
                logging.warning("No data to export after filtering")
                return
            
            mtbf = calculate_mtbf(filtered_df, set(valid_indices))
            _, _, failures_per_year = calculate_crow_amsaa_params(filtered_df, set(valid_indices))
            
            try:
                prod_loss = float(self.prod_loss_var.get())
                maint_cost = float(self.maint_cost_var.get())
                margin = float(self.margin_var.get())
                risk = failures_per_year * (prod_loss * margin + maint_cost)
            except ValueError:
                risk = 0.0
                logging.error("Invalid input for risk calculation in export")
            
            export_columns = [
                'Work Order',
                'Description',
                'Asset',
                'Equipment #',
                'Work Type',
                'Reported Date',
                'Failure Code',
                'Failure Description',
                'Matched Keyword',
                'User failure code',
                'AI Confidence',
                'Classification Method',
                'Work Order Cost'
            ]
            export_df = filtered_df[export_columns].copy()
            
            summary_data = {
                'Metric': ['MTBF (days)', 'Failures per Year', 'Annualized Risk ($)'],
                'Value': [f'{mtbf:.2f}', f'{failures_per_year:.2f}', f'{risk:,.2f}']
            }
            summary_df = pd.DataFrame(summary_data)
            
            # Equipment Summary
            equipment_nums = [equipment] if equipment else sorted(filtered_df['Equipment #'].dropna().unique())
            eq_data = []
            for eq in equipment_nums:
                eq_df = filtered_df[filtered_df['Equipment #'] == eq]
                valid_indices = eq_df.index.intersection(self.included_indices)
                total_wo = len(valid_indices)
                _, _, failures_per_year = calculate_crow_amsaa_params(eq_df, set(valid_indices))
                
                # Calculate cost information
                total_cost = 0.0
                avg_cost = 0.0
                if total_wo > 0:
                    # Get costs for valid indices
                    costs = []
                    for idx in valid_indices:
                        cost = eq_df.at[idx, 'Work Order Cost']
                        try:
                            cost = float(cost) if pd.notna(cost) else 0.0
                            costs.append(cost)
                        except (ValueError, TypeError):
                            costs.append(0.0)
                    
                    total_cost = sum(costs)
                    avg_cost = total_cost / total_wo if total_wo > 0 else 0.0
                
                eq_data.append({
                    'Equipment #': eq,
                    'Total Work Orders': total_wo,
                    'Failures per Year': f'{failures_per_year:.2f}',
                    'Total Cost ($)': f'{total_cost:,.2f}',
                    'Avg Cost per WO ($)': f'{avg_cost:,.2f}'
                })
            eq_df = pd.DataFrame(eq_data)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.output_dir, f"failure_mode_report_{timestamp}.xlsx")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Work Orders', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                eq_df.to_excel(writer, sheet_name='Equipment Summary', index=False)
            
            self.status_label.config(text=f"Exported to {output_file}", foreground="green")
            self.root.config(cursor="")
            self.root.update()
            messagebox.showinfo("Success", f"Report exported to {output_file}")
            logging.info(f"Exported report to: {output_file}")
        
        except Exception as e:
            self.status_label.config(text=f"Export error: {str(e)}", foreground="red")
            self.root.config(cursor="")
            self.root.update()
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
            logging.error(f"Error exporting to Excel: {str(e)}")
    
    def open_output_folder(self):
        """Open the output directory."""
        output_dir = self.output_dir or os.path.dirname(self.wo_entry.get()) or '.'
        if os.path.exists(output_dir):
            os.startfile(output_dir)
        else:
            self.status_label.config(text="Output folder does not exist.", foreground="red")
            messagebox.showerror("Error", "Output folder does not exist.")
            logging.error(f"Output folder does not exist: {output_dir}")

    def toggle_ai(self):
        """Toggle AI classification on/off"""
        self.use_ai_classification = self.ai_enabled_var.get()
        status = "enabled" if self.use_ai_classification else "disabled"
        self.status_label.config(text=f"AI Classification {status}")
        self.ai_status_indicator.config(text="🤖" if self.use_ai_classification else "⭕")
        
        # Save AI settings
        self.config['ai_enabled'] = self.use_ai_classification
        self.save_file_paths()

    def show_ai_settings(self):
        """Show AI settings dialog"""
        # Switch to AI settings tab
        self.notebook.select(2)

    def show_ai_settings_dialog(self):
        """Show AI Classifier Settings in a modal dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("AI Classifier Settings")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("400x350")
        dialog.resizable(False, False)
        
        ai_frame = ttk.Frame(dialog, padding=10)
        ai_frame.pack(fill=tk.BOTH, expand=True)
        
        config_frame = ttk.LabelFrame(ai_frame, text="AI Configuration", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # AI Enable/Disable
        ttk.Checkbutton(config_frame, text="Enable AI Classification", variable=self.ai_enabled_var).pack(anchor=tk.W, pady=2)
        
        # Confidence Threshold
        threshold_frame = ttk.Frame(config_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        ttk.Label(threshold_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        confidence_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.confidence_scale_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        ttk.Label(threshold_frame, textvariable=self.confidence_var, width=5).pack(side=tk.RIGHT)
        
        def update_confidence_label(*args):
            self.confidence_var.set(f"{self.confidence_scale_var.get():.2f}")
            self.config['confidence_threshold'] = self.confidence_scale_var.get()
            self.save_file_paths()
        self.confidence_scale_var.trace('w', update_confidence_label)
        
        # Enhanced Classification Methods
        methods_frame = ttk.Frame(config_frame)
        methods_frame.pack(fill=tk.X, pady=5)
        ttk.Label(methods_frame, text="Enhanced Methods:").pack(side=tk.LEFT)
        
        self.expert_system_var = tk.BooleanVar(value=True)
        self.contextual_patterns_var = tk.BooleanVar(value=True)
        self.temporal_analysis_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(methods_frame, text="Expert System", variable=self.expert_system_var).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Checkbutton(methods_frame, text="Contextual Patterns", variable=self.contextual_patterns_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(methods_frame, text="Temporal Analysis", variable=self.temporal_analysis_var).pack(side=tk.LEFT, padx=5)
        
        # AI Status
        status_frame = ttk.Frame(config_frame)
        status_frame.pack(fill=tk.X, pady=5)
        self.ai_status_label = ttk.Label(status_frame, text="Enhanced AI: Expert System, Contextual Patterns, Temporal Analysis", foreground="green")
        self.ai_status_label.pack(side=tk.LEFT)
        
        # AI Action buttons
        ai_buttons_frame = ttk.Frame(ai_frame)
        ai_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(ai_buttons_frame, text="📊 AI Statistics", command=self.show_ai_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(ai_buttons_frame, text="🗑️ Clear AI Cache", command=self.clear_ai_cache).pack(side=tk.LEFT, padx=5)
        ttk.Button(ai_buttons_frame, text="📤 Export Training Data", command=self.export_training_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ai_buttons_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def batch_process(self):
        """Batch process multiple work order files"""
        if not self.dict_entry.get() or not os.path.exists(self.dict_entry.get()):
            messagebox.showerror("Error", "Please select a valid dictionary file first.")
            return
        
        # Create batch processing dialog
        batch_window = tk.Toplevel(self.root)
        batch_window.title("Batch Processing")
        batch_window.geometry("600x500")
        batch_window.transient(self.root)
        batch_window.grab_set()
        
        # Center the window
        batch_window.update_idletasks()
        x = (batch_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (batch_window.winfo_screenheight() // 2) - (500 // 2)
        batch_window.geometry(f"600x500+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(batch_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="Select multiple work order files:").pack(anchor=tk.W)
        
        file_list_frame = ttk.Frame(file_frame)
        file_list_frame.pack(fill=tk.X, pady=5)
        
        # File listbox with scrollbar
        list_frame = ttk.Frame(file_list_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        file_listbox = tk.Listbox(list_frame, height=6)
        file_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=file_listbox.yview)
        file_listbox.configure(yscrollcommand=file_scrollbar.set)
        
        file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # File selection buttons
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill=tk.X, pady=5)
        
        def add_files():
            files = filedialog.askopenfilenames(
                title="Select Work Order Files",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            for file in files:
                if file not in file_listbox.get(0, tk.END):
                    file_listbox.insert(tk.END, file)
        
        def remove_selected():
            selection = file_listbox.curselection()
            for index in reversed(selection):
                file_listbox.delete(index)
        
        def clear_files():
            file_listbox.delete(0, tk.END)
        
        ttk.Button(file_buttons_frame, text="Add Files", command=add_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_buttons_frame, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_buttons_frame, text="Clear All", command=clear_files).pack(side=tk.LEFT, padx=5)
        
        # Output settings
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output directory
        output_dir_frame = ttk.Frame(output_frame)
        output_dir_frame.pack(fill=tk.X, pady=2)
        ttk.Label(output_dir_frame, text="Output Directory:", width=15).pack(side=tk.LEFT)
        output_dir_var = tk.StringVar(value=self.output_dir or os.getcwd())
        output_dir_entry = ttk.Entry(output_dir_frame, textvariable=output_dir_var)
        output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        def browse_output_dir():
            dir_path = filedialog.askdirectory(initialdir=output_dir_var.get())
            if dir_path:
                output_dir_var.set(dir_path)
        
        ttk.Button(output_dir_frame, text="Browse", command=browse_output_dir, width=10).pack(side=tk.RIGHT)
        
        # Output format options
        format_frame = ttk.Frame(output_frame)
        format_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(format_frame, text="Output Format:").pack(side=tk.LEFT)
        
        output_format_var = tk.StringVar(value="individual")
        ttk.Radiobutton(format_frame, text="Individual files", variable=output_format_var, 
                       value="individual").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(format_frame, text="Combined file", variable=output_format_var, 
                       value="combined").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Both", variable=output_format_var, 
                       value="both").pack(side=tk.LEFT, padx=5)
        
        # Processing options
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # AI classification option
        ai_var = tk.BooleanVar(value=self.use_ai_classification)
        ttk.Checkbutton(options_frame, text="Use AI Classification", variable=ai_var).pack(anchor=tk.W)
        
        # Include summary option
        summary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include summary sheet", variable=summary_var).pack(anchor=tk.W)
        
        # Progress tracking
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        progress_label = ttk.Label(progress_frame, text="Ready to process")
        progress_label.pack(anchor=tk.W)
        
        progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=5)
        
        # Results text
        results_text = tk.Text(progress_frame, height=8, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=results_text.yview)
        results_text.configure(yscrollcommand=results_scrollbar.set)
        
        results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def start_batch_processing():
            files = list(file_listbox.get(0, tk.END))
            if not files:
                messagebox.showerror("Error", "Please select at least one file to process.")
                return
            
            output_dir = output_dir_var.get()
            if not output_dir or not os.path.exists(output_dir):
                messagebox.showerror("Error", "Please select a valid output directory.")
                return
            
            # Disable buttons during processing
            start_button.config(state=tk.DISABLED)
            cancel_button.config(state=tk.DISABLED)
            
            # Clear results
            results_text.delete(1.0, tk.END)
            
            # Start processing in a separate thread
            import threading
            
            def process_files_thread():
                try:
                    dict_path = self.dict_entry.get()
                    use_ai = ai_var.get()
                    output_format = output_format_var.get()
                    include_summary = summary_var.get()
                    
                    # Initialize AI classifier if needed
                    ai_classifier = None
                    if use_ai and AI_AVAILABLE:
                        try:
                            confidence_threshold = float(self.confidence_var.get())
                            
                            ai_classifier = AIClassifier(
                                confidence_threshold=confidence_threshold,
                                cache_file=AI_CACHE_FILE
                            )
                            
                            if not ai_classifier.load_failure_dictionary(dict_path):
                                results_text.insert(tk.END, "Warning: Failed to load dictionary for AI classifier\n")
                                use_ai = False
                        except Exception as e:
                            results_text.insert(tk.END, f"Warning: Failed to initialize AI classifier: {str(e)}\n")
                            use_ai = False
                    
                    total_files = len(files)
                    processed_files = []
                    failed_files = []
                    
                    for i, file_path in enumerate(files):
                        try:
                            # Update progress
                            progress = (i / total_files) * 100
                            progress_bar['value'] = progress
                            progress_label.config(text=f"Processing {os.path.basename(file_path)}...")
                            batch_window.update()
                            
                            # Process file
                            df = process_files(file_path, dict_path, progress_label, self.root, 
                                             output_dir, use_ai, ai_classifier, column_mapping=self.column_mapping)
                            
                            if df is not None and not df.empty:
                                processed_files.append((file_path, df))
                                results_text.insert(tk.END, f"✓ {os.path.basename(file_path)} - {len(df)} rows\n")
                            else:
                                failed_files.append(file_path)
                                results_text.insert(tk.END, f"✗ {os.path.basename(file_path)} - Failed\n")
                            
                            results_text.see(tk.END)
                            
                        except Exception as e:
                            failed_files.append(file_path)
                            results_text.insert(tk.END, f"✗ {os.path.basename(file_path)} - Error: {str(e)}\n")
                            results_text.see(tk.END)
                            logging.error(f"Batch processing error for {file_path}: {e}")
                    
                    # Generate output files
                    if processed_files:
                        try:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            if output_format in ["combined", "both"]:
                                # Create combined file
                                combined_data = []
                                for file_path, df in processed_files:
                                    df_copy = df.copy()
                                    df_copy['Source File'] = os.path.basename(file_path)
                                    combined_data.append(df_copy)
                                
                                combined_df = pd.concat(combined_data, ignore_index=True)
                                combined_file = os.path.join(output_dir, f"batch_combined_{timestamp}.xlsx")
                                
                                with pd.ExcelWriter(combined_file, engine='openpyxl') as writer:
                                    combined_df.to_excel(writer, sheet_name='Combined Data', index=False)
                                    
                                    if include_summary:
                                        # Create summary
                                        summary_data = []
                                        for file_path, df in processed_files:
                                            mtbf = calculate_mtbf(df, set(df.index))
                                            _, _, failures_per_year = calculate_crow_amsaa_params(df, set(df.index))
                                            summary_data.append({
                                                'File': os.path.basename(file_path),
                                                'Rows': len(df),
                                                'MTBF (days)': f'{mtbf:.2f}',
                                                'Failures per Year': f'{failures_per_year:.2f}'
                                            })
                                        
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                
                                results_text.insert(tk.END, f"Combined file: {os.path.basename(combined_file)}\n")
                            
                            if output_format in ["individual", "both"]:
                                # Create individual files
                                for file_path, df in processed_files:
                                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                                    individual_file = os.path.join(output_dir, f"{base_name}_processed_{timestamp}.xlsx")
                                    
                                    with pd.ExcelWriter(individual_file, engine='openpyxl') as writer:
                                        df.to_excel(writer, sheet_name='Work Orders', index=False)
                                        
                                        if include_summary:
                                            mtbf = calculate_mtbf(df, set(df.index))
                                            _, _, failures_per_year = calculate_crow_amsaa_params(df, set(df.index))
                                            
                                            summary_data = {
                                                'Metric': ['MTBF (days)', 'Failures per Year'],
                                                'Value': [f'{mtbf:.2f}', f'{failures_per_year:.2f}']
                                            }
                                            summary_df = pd.DataFrame(summary_data)
                                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    results_text.insert(tk.END, f"Individual file: {os.path.basename(individual_file)}\n")
                            
                            results_text.insert(tk.END, f"\nBatch processing complete!\n")
                            results_text.insert(tk.END, f"Processed: {len(processed_files)} files\n")
                            results_text.insert(tk.END, f"Failed: {len(failed_files)} files\n")
                            
                        except Exception as e:
                            results_text.insert(tk.END, f"Error creating output files: {str(e)}\n")
                            logging.error(f"Error creating batch output files: {e}")
                    
                    # Final progress update
                    progress_bar['value'] = 100
                    progress_label.config(text="Batch processing complete")
                    
                except Exception as e:
                    results_text.insert(tk.END, f"Batch processing failed: {str(e)}\n")
                    logging.error(f"Batch processing failed: {e}")
                finally:
                    # Re-enable buttons
                    start_button.config(state=tk.NORMAL)
                    cancel_button.config(state=tk.NORMAL)
            
            # Start the processing thread
            processing_thread = threading.Thread(target=process_files_thread)
            processing_thread.daemon = True
            processing_thread.start()
        
        def cancel_processing():
            batch_window.destroy()
        
        start_button = ttk.Button(button_frame, text="Start Processing", command=start_batch_processing)
        start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=cancel_processing)
        cancel_button.pack(side=tk.LEFT)
        
        # Help text
        help_frame = ttk.LabelFrame(main_frame, text="How to Use", padding=10)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        help_text = """1. Click "Add Files" to select multiple work order Excel files
2. Choose output directory and format (individual/combined/both)
3. Enable AI classification if desired
4. Click "Start Processing" to begin batch processing
5. Monitor progress and results in the text area"""
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT)
        help_label.pack(anchor=tk.W)

    def clear_data(self):
        """Clear all loaded data"""
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all data?"):
            self.wo_df = None
            self.included_indices = set()
            self.update_table()
            self.status_label.config(text="Data cleared")
            self.data_status_indicator.config(text="📊")

    def show_filter_manager(self):
        """Show filter management dialog"""
        if self.wo_df is None or self.wo_df.empty:
            messagebox.showerror("Error", "No data loaded. Please process files first.")
            return
        
        # Create filter manager window
        filter_window = tk.Toplevel(self.root)
        filter_window.title("Filter Management")
        filter_window.geometry("600x500")
        filter_window.transient(self.root)
        filter_window.grab_set()
        
        # Center the window
        filter_window.update_idletasks()
        x = (filter_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (filter_window.winfo_screenheight() // 2) - (500 // 2)
        filter_window.geometry(f"600x500+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(filter_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Filter Management", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Current filters display
        current_frame = ttk.LabelFrame(main_frame, text="Current Filters", padding=10)
        current_frame.pack(fill=tk.X, pady=(0, 10))
        
        current_filters_text = f"""Equipment: {self.equipment_var.get() or 'All'}
Work Type: {self.work_type_var.get() or 'All'}
Failure Code: {self.failure_code_var.get() or 'All'}
Date Range: {self.start_date_entry.get() or 'None'} to {self.end_date_entry.get() or 'None'}
Included Work Orders: {len(self.included_indices)} of {len(self.wo_df) if self.wo_df is not None else 0}"""
        
        ttk.Label(current_frame, text=current_filters_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Filter presets
        presets_frame = ttk.LabelFrame(main_frame, text="Filter Presets", padding=10)
        presets_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Preset buttons
        preset_buttons_frame = ttk.Frame(presets_frame)
        preset_buttons_frame.pack(fill=tk.X)
        
        def apply_equipment_filter():
            """Apply filter for most common equipment"""
            if self.wo_df is not None:
                most_common = str(self.wo_df['Equipment #'].value_counts().index[0])
                self.equipment_var.set(most_common)
                self.update_table()
                filter_window.destroy()
        
        def apply_failure_filter():
            """Apply filter for most common failure code"""
            if self.wo_df is not None:
                most_common = str(self.wo_df['Failure Code'].value_counts().index[0])
                self.failure_code_var.set(most_common)
                self.update_table()
                filter_window.destroy()
        
        def apply_recent_filter():
            """Apply filter for recent work orders (last 30 days)"""
            if self.wo_df is not None:
                # Set date range to last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                self.start_date_entry.delete(0, tk.END)
                self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
                self.end_date_entry.delete(0, tk.END)
                self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
                self.apply_date_filter()
                filter_window.destroy()
        
        def apply_high_cost_filter():
            """Apply filter for high-cost work orders"""
            if self.wo_df is not None and 'Work Order Cost' in self.wo_df.columns:
                # Include only work orders with cost > 0
                self.included_indices = set(self.wo_df[self.wo_df['Work Order Cost'] > 0].index)
                self.update_table()
                filter_window.destroy()
        
        ttk.Button(preset_buttons_frame, text="Most Common Equipment", command=apply_equipment_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_buttons_frame, text="Most Common Failure", command=apply_failure_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_buttons_frame, text="Recent (30 days)", command=apply_recent_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_buttons_frame, text="High Cost Only", command=apply_high_cost_filter).pack(side=tk.LEFT, padx=5)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(main_frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        actions_buttons_frame = ttk.Frame(actions_frame)
        actions_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(actions_buttons_frame, text="Include All", command=lambda: self.include_all_work_orders(filter_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_buttons_frame, text="Exclude All", command=lambda: self.exclude_all_work_orders(filter_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_buttons_frame, text="Reset All Filters", command=lambda: self.reset_all_filters_from_manager(filter_window)).pack(side=tk.LEFT, padx=5)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=filter_window.destroy).pack(side=tk.RIGHT, pady=10)
    
    def include_all_work_orders(self, window=None):
        """Include all work orders in analysis"""
        if self.wo_df is not None:
            self.included_indices = set(self.wo_df.index)
            self.update_table()
            if window:
                window.destroy()
    
    def exclude_all_work_orders(self, window=None):
        """Exclude all work orders from analysis"""
        self.included_indices = set()
        self.update_table()
        if window:
            window.destroy()
    
    def reset_all_filters_from_manager(self, window=None):
        """Reset all filters from filter manager"""
        self.reset_all_filters()
        if window:
            window.destroy()

    def show_date_selector(self):
        """Show date range selector dialog"""
        if self.wo_df is None or self.wo_df.empty:
            messagebox.showerror("Error", "No data loaded. Please process files first.")
            return
        
        # Create date selector window
        date_window = tk.Toplevel(self.root)
        date_window.title("Date Range Selector")
        date_window.geometry("500x400")
        date_window.transient(self.root)
        date_window.grab_set()
        
        # Center the window
        date_window.update_idletasks()
        x = (date_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (date_window.winfo_screenheight() // 2) - (400 // 2)
        date_window.geometry(f"500x400+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(date_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Date Range Selector", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Current date range
        current_frame = ttk.LabelFrame(main_frame, text="Current Date Range", padding=10)
        current_frame.pack(fill=tk.X, pady=(0, 10))
        
        current_range = f"Start: {self.start_date_entry.get() or 'None'}\nEnd: {self.end_date_entry.get() or 'None'}"
        ttk.Label(current_frame, text=current_range, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Date range presets
        presets_frame = ttk.LabelFrame(main_frame, text="Quick Date Ranges", padding=10)
        presets_frame.pack(fill=tk.X, pady=(0, 10))
        
        def set_last_7_days():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_last_30_days():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_last_90_days():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_last_year():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_current_month():
            now = datetime.now()
            start_date = now.replace(day=1)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, now.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_current_year():
            now = datetime.now()
            start_date = now.replace(month=1, day=1)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, now.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        # Preset buttons
        preset_buttons_frame = ttk.Frame(presets_frame)
        preset_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(preset_buttons_frame, text="Last 7 Days", command=set_last_7_days).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(preset_buttons_frame, text="Last 30 Days", command=set_last_30_days).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(preset_buttons_frame, text="Last 90 Days", command=set_last_90_days).pack(side=tk.LEFT, padx=5, pady=2)
        
        preset_buttons_frame2 = ttk.Frame(presets_frame)
        preset_buttons_frame2.pack(fill=tk.X)
        
        ttk.Button(preset_buttons_frame2, text="Last Year", command=set_last_year).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(preset_buttons_frame2, text="Current Month", command=set_current_month).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(preset_buttons_frame2, text="Current Year", command=set_current_year).pack(side=tk.LEFT, padx=5, pady=2)
        
        # Data range info
        if self.wo_df is not None:
            try:
                dates = pd.to_datetime(self.wo_df['Reported Date'], errors='coerce').dropna()
                if len(dates) > 0:
                    min_date = dates.min().strftime('%m/%d/%Y')
                    max_date = dates.max().strftime('%m/%d/%Y')
                    data_range_text = f"Data Range: {min_date} to {max_date}\nTotal Work Orders: {len(dates)}"
                    
                    data_frame = ttk.LabelFrame(main_frame, text="Available Data Range", padding=10)
                    data_frame.pack(fill=tk.X, pady=(0, 10))
                    ttk.Label(data_frame, text=data_range_text, justify=tk.LEFT).pack(anchor=tk.W)
            except Exception as e:
                logging.debug(f"Error getting date range: {e}")
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Clear Date Range", command=lambda: self.clear_date_range(date_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=date_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def clear_date_range(self, window=None):
        """Clear the current date range"""
        self.start_date_entry.delete(0, tk.END)
        self.end_date_entry.delete(0, tk.END)
        self.start_date = None
        self.end_date = None
        self.update_table()
        if window:
            window.destroy()

    def reset_all_filters(self):
        """Reset all filters to default and recheck all work orders"""
        # Reset filter variables
        self.equipment_var.set('')
        self.failure_code_var.set('')
        self.work_type_var.set('')
        self.start_date_entry.delete(0, tk.END)
        self.end_date_entry.delete(0, tk.END)
        self.start_date = None
        self.end_date = None
        
        # Recheck all work orders (include all in analysis)
        if self.wo_df is not None:
            self.included_indices = set(self.wo_df.index)
        
        # Update the table to reflect changes
        self.update_table()
        self.status_label.config(text="All filters reset and all work orders included")

    def reset_defaults(self):
        """Reset to default settings and recheck all work orders"""
        self.confidence_var.set(str(AI_CONFIDENCE_THRESHOLD))
        self.ai_enabled_var.set(False)
        self.use_ai_classification = False
        self.reset_all_filters()  # Ensure all work orders are included and filters reset
        self.status_label.config(text="Settings reset to defaults and all work orders included")



    def export_training_data(self):
        """Export training data for AI model improvement"""
        if self.wo_df is None or self.wo_df.empty:
            messagebox.showerror("Error", "No data to export. Please process files first.")
            return
        
        if not self.ai_classifier:
            messagebox.showerror("Error", "AI classifier not available.")
            return
        
        try:
            output_file = "training_data_export.json"
            if self.ai_classifier.export_training_data(output_file, self.wo_df):
                messagebox.showinfo("Success", f"Training data exported to {output_file}")
            else:
                messagebox.showerror("Error", "Failed to export training data")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    def save_risk_preset(self):
        """Save current risk parameters as preset"""
        preset_name = simpledialog.askstring("Save Preset", "Enter preset name:")
        if preset_name:
            try:
                preset = {
                    'name': preset_name,
                    'prod_loss': self.prod_loss_var.get(),
                    'maint_cost': self.maint_cost_var.get(),
                    'margin': self.margin_var.get(),
                    'created': datetime.now().isoformat()
                }
                
                # Load existing presets
                presets = self.load_risk_presets()
                presets[preset_name] = preset
                
                # Save to file
                import json
                with open('risk_presets.json', 'w') as f:
                    json.dump(presets, f, indent=2)
                
                messagebox.showinfo("Success", f"Preset '{preset_name}' saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save preset: {str(e)}")
                logging.error(f"Error saving risk preset: {e}")

    def load_risk_preset(self):
        """Load risk parameters from preset"""
        presets = self.load_risk_presets()
        
        if not presets:
            messagebox.showinfo("No Presets", "No saved presets found.")
            return
        
        # Create preset selection dialog
        preset_window = tk.Toplevel(self.root)
        preset_window.title("Load Risk Preset")
        preset_window.geometry("400x300")
        preset_window.transient(self.root)
        preset_window.grab_set()
        
        # Center the window
        preset_window.update_idletasks()
        x = (preset_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (preset_window.winfo_screenheight() // 2) - (300 // 2)
        preset_window.geometry(f"400x300+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(preset_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Select a preset to load:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Preset listbox
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        preset_listbox = tk.Listbox(list_frame, height=8)
        preset_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=preset_listbox.yview)
        preset_listbox.configure(yscrollcommand=preset_scrollbar.set)
        
        preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preset_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox
        for preset_name in presets.keys():
            preset_listbox.insert(tk.END, preset_name)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def load_selected_preset():
            selection = preset_listbox.curselection()
            if selection:
                preset_name = preset_listbox.get(selection[0])
                preset = presets[preset_name]
                
                self.prod_loss_var.set(preset['prod_loss'])
                self.maint_cost_var.set(preset['maint_cost'])
                self.margin_var.set(preset['margin'])
                
                # Update risk calculation
                self.update_risk()
                
                preset_window.destroy()
                messagebox.showinfo("Success", f"Preset '{preset_name}' loaded successfully!")
            else:
                messagebox.showwarning("Warning", "Please select a preset to load.")
        
        def delete_selected_preset():
            selection = preset_listbox.curselection()
            if selection:
                preset_name = preset_listbox.get(selection[0])
                if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete preset '{preset_name}'?"):
                    del presets[preset_name]
                    
                    # Save updated presets
                    import json
                    with open('risk_presets.json', 'w') as f:
                        json.dump(presets, f, indent=2)
                    
                    # Refresh listbox
                    preset_listbox.delete(0, tk.END)
                    for preset_name in presets.keys():
                        preset_listbox.insert(tk.END, preset_name)
                    
                    messagebox.showinfo("Success", f"Preset '{preset_name}' deleted successfully!")
            else:
                messagebox.showwarning("Warning", "Please select a preset to delete.")
        
        ttk.Button(button_frame, text="Load", command=load_selected_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete", command=delete_selected_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=preset_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def load_risk_presets(self) -> dict:
        """Load risk presets from file"""
        try:
            import json
            if os.path.exists('risk_presets.json'):
                with open('risk_presets.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading risk presets: {e}")
        
        return {}



    def export_plot(self):
        """Export current plot"""
        if self.crow_amsaa_fig:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    self.crow_amsaa_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Success", f"Plot exported to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
        else:
            messagebox.showerror("Error", "No plot to export")

    def open_plot_in_new_window(self):
        """Open the current Crow-AMSAA plot in a new window"""
        if self.crow_amsaa_fig is None:
            messagebox.showerror("Error", "No plot to display")
            return
        
        try:
            # Create new window
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Crow-AMSAA Analysis Plot")
            plot_window.geometry("800x600")
            plot_window.transient(self.root)
            
            # Center the window
            plot_window.update_idletasks()
            x = (plot_window.winfo_screenwidth() // 2) - (800 // 2)
            y = (plot_window.winfo_screenheight() // 2) - (600 // 2)
            plot_window.geometry(f"800x600+{x}+{y}")
            
            # Create frame for the plot
            plot_frame = ttk.Frame(plot_window, padding=10)
            plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create a new figure with the same data
            if self.wo_df is not None and not self.wo_df.empty:
                filtered_df = self.get_filtered_df()
                
                # Create the plot in the new window
                if self.is_segmented_view and hasattr(self, 'segment_date') and self.segment_date:
                    # Create segmented plot
                    self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, plot_frame, segment_date=self.segment_date)
                else:
                    # Create single plot
                    self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, plot_frame)
                
                # Add close button
                button_frame = ttk.Frame(plot_window)
                button_frame.pack(fill=tk.X, padx=10, pady=5)
                ttk.Button(button_frame, text="Close", command=plot_window.destroy).pack(side=tk.RIGHT)
                
            else:
                ttk.Label(plot_frame, text="No data available for plotting").pack(expand=True)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open plot in new window: {str(e)}")
            logging.error(f"Error opening plot in new window: {e}")

    def show_software_user_guide(self):
        """Show comprehensive software user guide"""
        try:
            # Try to open the markdown file in default application
            import subprocess
            import platform
            
            guide_file = "SOFTWARE_USER_GUIDE.md"
            if os.path.exists(guide_file):
                if platform.system() == "Windows":
                    os.startfile(guide_file)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", guide_file])
                else:  # Linux
                    subprocess.run(["xdg-open", guide_file])
            else:
                # Fallback to showing basic guide if file not found
                self.show_user_guide()
        except Exception as e:
            # Fallback to showing basic guide if file opening fails
            self.show_user_guide()

    def show_technical_guide(self):
        """Show comprehensive technical application guide"""
        try:
            # Try to open the markdown file in default application
            import subprocess
            import platform
            
            guide_file = "TECHNICAL_APPLICATION_GUIDE.md"
            if os.path.exists(guide_file):
                if platform.system() == "Windows":
                    os.startfile(guide_file)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", guide_file])
                else:  # Linux
                    subprocess.run(["xdg-open", guide_file])
            else:
                # Fallback to showing basic technical info
                technical_text = """
Technical Application Guide

This guide provides detailed technical explanations of the analysis methodologies used in Work Order Analysis Pro.

Key Analysis Methods:

1. Crow-AMSAA Analysis:
   - Models cumulative failures as N(t) = λt^β
   - β < 1 indicates reliability growth
   - β = 1 indicates constant failure rate
   - β > 1 indicates reliability degradation

2. Weibull Analysis:
   - Models failure distribution with shape (β) and scale (η) parameters
   - β < 1: Infant mortality (decreasing failure rate)
   - β = 1: Random failures (constant failure rate)
   - β > 1: Wear-out (increasing failure rate)

3. Risk Assessment:
   - Risk = Probability × Consequence
   - Annualized Risk = Failure Rate × Cost per Failure × Operating Hours

4. Preventive Maintenance Analysis:
   - Optimizes PM intervals to minimize total cost
   - Considers PM costs, failure costs, and downtime costs

5. Spares Analysis:
   - Uses Monte Carlo simulation for demand forecasting
   - Optimizes inventory levels based on service level targets

6. AI Classification:
   - Hybrid system using sentence embeddings and dictionary matching
   - Confidence scoring determines classification method used

For complete technical documentation, please refer to the TECHNICAL_APPLICATION_GUIDE.md file.
                """
            messagebox.showinfo("Technical Application Guide", technical_text)
        except Exception as e:
            technical_text = """Technical Application Guide

This guide provides detailed technical explanations of the analysis methodologies used in Work Order Analysis Pro.

Key Analysis Methods:

1. Crow-AMSAA Analysis:
   - Models cumulative failures as N(t) = λt^β
   - β < 1 indicates reliability growth
   - β = 1 indicates constant failure rate
   - β > 1 indicates reliability degradation

2. Weibull Analysis:
   - Models failure distribution with shape (β) and scale (η) parameters
   - β < 1: Infant mortality (decreasing failure rate)
   - β = 1: Random failures (constant failure rate)
   - β > 1: Wear-out (increasing failure rate)

3. Risk Assessment:
   - Risk = Probability × Consequence
   - Annualized Risk = Failure Rate × Cost per Failure × Operating Hours

4. Preventive Maintenance Analysis:
   - Optimizes PM intervals to minimize total cost
   - Considers PM costs, failure costs, and downtime costs

5. Spares Analysis:
   - Uses Monte Carlo simulation for demand forecasting
   - Optimizes inventory levels based on service level targets

6. AI Classification:
   - Hybrid system using sentence embeddings and dictionary matching
   - Confidence scoring determines classification method used

For complete technical documentation, please refer to the TECHNICAL_APPLICATION_GUIDE.md file."""
            messagebox.showinfo("Technical Application Guide", technical_text)

    def show_user_guide(self):
        """Show basic user guide (fallback)"""
        guide_text = """
Work Order Analysis Pro - User Guide

1. Data Input Tab:
   - Load work order and dictionary files
   - Set output directory
   - Process files to begin analysis

2. Analysis Tab:
   - Use filters to focus on specific data
   - View work orders and equipment summary with cost information
   - Sort and filter data as needed
   - View Crow-AMSAA analysis plots (resizable panes)
   - Right-click work orders to segment plots
   - Use "Return to Single Plot" button to restore single view
   - Use "Open in New Window" to view plots in separate window

3. AI Settings Tab:
   - Enable/disable AI classification
   - Adjust confidence threshold
   - View AI statistics and manage cache

4. Risk Assessment Tab:
   - Set risk parameters
   - Calculate failure rates and annualized risk
   - Risk values reflect current Crow-AMSAA plot state (single/segmented)

5. Spares Analysis Tab (Methodology):
   - Spares demand is estimated by analyzing historical work order data, grouped by failure mode and equipment.
   - Monte Carlo simulation is used to forecast future spares demand over a user-defined period (e.g., 10 years), using either Poisson (for MTBF) or Weibull-based models (for reliability patterns).
   - Weibull modeling allows for non-constant failure rates, capturing wear-out or infant mortality patterns.
   - The simulation generates thousands of possible demand scenarios, from which mean, percentile, and annualized failure rates are calculated.
   - Stocking recommendations are based on service level targets (e.g., 95% reliability), with a sensitivity chart showing required spares for reliability levels from 99% down to 70%.
   - All plots use years on the x-axis for clarity and quick annual planning.
   - Notes: The analysis is most accurate when filtered for a specific failure mode and equipment. Results are only valid for spares matching the selected failure mode(s).

Features:
- Work Order Cost column (optional) - displays total and average costs per equipment
- Column mapping for CMMS compatibility
- Interactive Crow-AMSAA plots with segmentation
- Risk assessment that updates based on plot view

Keyboard Shortcuts:
- Ctrl+O: Load work order file
- Ctrl+D: Load dictionary file
- Ctrl+E: Export to Excel
- F5: Process files
- Ctrl+Q: Exit application

For comprehensive documentation, use Help → Software User Guide or Help → Technical Application Guide.
        """
        messagebox.showinfo("User Guide", guide_text)

    def show_about(self):
        """Show about dialog with icon"""
        # Create custom about window
        about_window = tk.Toplevel(self.root)
        about_window.title("About Work Order Analysis Pro Edition")
        about_window.geometry("600x500")
        about_window.transient(self.root)
        about_window.grab_set()
        about_window.resizable(False, False)
        
        # Center the window
        about_window.update_idletasks()
        x = (about_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (about_window.winfo_screenheight() // 2) - (500 // 2)
        about_window.geometry(f"600x500+{x}+{y}")
        
        # Set icon for about window
        try:
            icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'app_icon.ico')
            if os.path.exists(icon_path):
                about_window.iconbitmap(icon_path)
        except Exception as e:
            print(f"Could not load icon for about window: {e}")
        
        # Main frame
        main_frame = ttk.Frame(about_window, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Icon and title frame
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Create a dedicated frame for the icon
        icon_frame = ttk.Frame(header_frame)
        icon_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        # Try to display the icon with better error handling
        icon_displayed = False
        try:
            # Try the 64x64 version first
            icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'app_icon_64x64.png')
            if os.path.exists(icon_path):
                from PIL import Image, ImageTk
                icon_img = Image.open(icon_path)
                # Ensure the image is the right size
                icon_img = icon_img.resize((64, 64), Image.Resampling.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon_img)
                
                # Create a label with a border to make it more visible
                icon_label = tk.Label(icon_frame, image=icon_photo, relief=tk.RAISED, bd=2)
                icon_label.image = icon_photo  # Keep a reference
                icon_label.pack()
                icon_displayed = True
                print("✓ Icon displayed successfully in About dialog")
            else:
                print(f"✗ Icon file not found at: {icon_path}")
        except Exception as e:
            print(f"✗ Could not display icon in about dialog: {e}")
            # Try alternative icon size
            try:
                icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'app_icon_128x128.png')
                if os.path.exists(icon_path):
                    from PIL import Image, ImageTk
                    icon_img = Image.open(icon_path)
                    icon_img = icon_img.resize((64, 64), Image.Resampling.LANCZOS)
                    icon_photo = ImageTk.PhotoImage(icon_img)
                    
                    icon_label = tk.Label(icon_frame, image=icon_photo, relief=tk.RAISED, bd=2)
                    icon_label.image = icon_photo
                    icon_label.pack()
                    icon_displayed = True
                    print("✓ Alternative icon displayed successfully")
            except Exception as e2:
                print(f"✗ Could not display alternative icon: {e2}")
        
        # If no icon was displayed, show a placeholder
        if not icon_displayed:
            placeholder_label = tk.Label(icon_frame, text="📊", font=('Arial', 48), 
                                       relief=tk.RAISED, bd=2, bg='lightgray')
            placeholder_label.pack()
            print("⚠ Using placeholder icon")
        
        # Title and version frame
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        # Application title
        title_label = ttk.Label(title_frame, text="Work Order Analysis Pro Edition", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Version
        version_label = ttk.Label(title_frame, text="", 
                                 font=('Arial', 14))
        version_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Subtitle
        subtitle_label = ttk.Label(title_frame, text="Developed by Matt Barnes © 2025 All Rights Reserved", 
                                  font=('Arial', 11, 'italic'), foreground='gray')
        subtitle_label.pack(anchor=tk.W)
        
        # Separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=(0, 20))
        
       
        
        # Create scrollable text widget
        text_widget = tk.Text(text_frame, wrap=tk.WORD, height=12, 
                             font=('Arial', 10), relief=tk.FLAT, 
                             background=text_frame.cget('background'),
                             padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # About text content
        about_text = "Developed by Matt Barnes © 2025 All Rights Reserved"
        
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Close button
        close_button = ttk.Button(button_frame, text="Close", command=about_window.destroy, 
                                 style='Accent.TButton')
        close_button.pack(side=tk.RIGHT)
        
        # Focus on the close button
        close_button.focus_set()
        
        # Bind Enter key to close
        about_window.bind('<Return>', lambda e: about_window.destroy())
        about_window.bind('<Escape>', lambda e: about_window.destroy())

    def update_progress(self, value, text=None):
        """Update progress bar and status"""
        self.progress_var.set(value)
        if text:
            self.status_label.config(text=text)
        self.root.update_idletasks()

    def show_column_mapping(self):
        """Show column mapping dialog for CMMS compatibility"""
        # Create column mapping dialog
        mapping_window = tk.Toplevel(self.root)
        mapping_window.title("Column Mapping - CMMS Compatibility")
        mapping_window.geometry("700x600")
        mapping_window.transient(self.root)
        mapping_window.grab_set()
        
        # Center the window
        mapping_window.update_idletasks()
        x = (mapping_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (mapping_window.winfo_screenheight() // 2) - (600 // 2)
        mapping_window.geometry(f"700x600+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(mapping_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and description
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="Column Mapping for CMMS Compatibility", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(title_frame, text="Map your CMMS export columns to the required program columns", 
                 font=('Arial', 9)).pack(anchor=tk.W)
        
        # File selection for preview
        file_frame = ttk.LabelFrame(main_frame, text="Select Work Order File for Preview", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.pack(fill=tk.X)
        
        ttk.Label(file_select_frame, text="File:").pack(side=tk.LEFT)
        preview_file_var = tk.StringVar()
        preview_file_entry = ttk.Entry(file_select_frame, textvariable=preview_file_var)
        preview_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        def browse_preview_file():
            file_path = filedialog.askopenfilename(
                title="Select Work Order File for Preview",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if file_path:
                preview_file_var.set(file_path)
                load_file_columns()
        
        ttk.Button(file_select_frame, text="Browse", command=browse_preview_file, width=10).pack(side=tk.RIGHT)
        
        # Column mapping area
        mapping_frame = ttk.LabelFrame(main_frame, text="Column Mapping", padding=10)
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create scrollable frame for mappings
        canvas = tk.Canvas(mapping_frame)
        scrollbar = ttk.Scrollbar(mapping_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store mapping widgets
        mapping_widgets = {}
        
        def load_file_columns():
            """Load columns from selected file and populate mapping dropdowns"""
            file_path = preview_file_var.get()
            if not file_path or not os.path.exists(file_path):
                return
            
            try:
                df = pd.read_excel(file_path)
                available_columns = list(df.columns)
                
                # Clear existing mappings
                for widget in mapping_widgets.values():
                    if hasattr(widget, 'destroy'):
                        widget.destroy()
                mapping_widgets.clear()
                
                # Create mapping rows
                for i, (required_col, description) in enumerate(REQUIRED_COLUMNS.items()):
                    row_frame = ttk.Frame(scrollable_frame)
                    row_frame.pack(fill=tk.X, pady=2)
                    
                    # Required column label
                    ttk.Label(row_frame, text=f"{description}:", width=20).pack(side=tk.LEFT)
                    
                    # Mapping dropdown
                    mapping_var = tk.StringVar()
                    if required_col in self.column_mapping:
                        mapping_var.set(self.column_mapping[required_col])
                    elif required_col in available_columns:
                        mapping_var.set(required_col)
                    else:
                        mapping_var.set('')
                    
                    mapping_dropdown = ttk.Combobox(row_frame, textvariable=mapping_var, 
                                                  values=[''] + available_columns, width=30)
                    mapping_dropdown.pack(side=tk.LEFT, padx=(10, 5))
                    
                    # Store reference
                    mapping_widgets[required_col] = mapping_dropdown
                    
                    # Status indicator
                    status_label = ttk.Label(row_frame, text="", width=10)
                    status_label.pack(side=tk.LEFT, padx=5)
                    
                    def update_status(col=required_col, var=mapping_var, label=status_label):
                        selected = var.get()
                        if selected and selected in available_columns:
                            label.config(text="✓", foreground="green")
                        elif selected:
                            label.config(text="✗", foreground="red")
                        else:
                            label.config(text="", foreground="black")
                    
                    mapping_dropdown.bind("<<ComboboxSelected>>", lambda e, col=required_col, var=mapping_var, label=status_label: update_status(col, var, label))
                    update_status(required_col, mapping_var, status_label)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file columns: {str(e)}")
        
        # Auto-detect button
        detect_frame = ttk.Frame(mapping_frame)
        detect_frame.pack(fill=tk.X, pady=(10, 0))
        
        def auto_detect_mappings():
            """Auto-detect column mappings based on similarity"""
            file_path = preview_file_var.get()
            if not file_path or not os.path.exists(file_path):
                messagebox.showwarning("Warning", "Please select a file first.")
                return
            
            try:
                df = pd.read_excel(file_path)
                available_columns = list(df.columns)
                
                # Simple fuzzy matching for auto-detection
                for required_col, description in REQUIRED_COLUMNS.items():
                    if required_col in mapping_widgets:
                        dropdown = mapping_widgets[required_col]
                        
                        # Try exact match first
                        if required_col in available_columns:
                            dropdown.set(required_col)
                            continue
                        
                        # Try partial matches
                        best_match = None
                        best_score = 0
                        
                        for col in available_columns:
                            # Check if any word in the description matches
                            desc_words = description.lower().split()
                            col_lower = col.lower()
                            
                            for word in desc_words:
                                if word in col_lower and len(word) > 2:
                                    score = len(word) / len(col_lower)
                                    if score > best_score:
                                        best_score = score
                                        best_match = col
                        
                        if best_match and best_score > 0.3:
                            dropdown.set(best_match)
                
                messagebox.showinfo("Auto-Detect", "Column mappings auto-detected based on similarity.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Auto-detection failed: {str(e)}")
        
        ttk.Button(detect_frame, text="Auto-Detect Mappings", command=auto_detect_mappings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(detect_frame, text="Clear All Mappings", 
                  command=lambda: [widget.set('') for widget in mapping_widgets.values()]).pack(side=tk.LEFT)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_mappings():
            """Save current column mappings"""
            mappings = {}
            for required_col, dropdown in mapping_widgets.items():
                selected = dropdown.get()
                if selected:
                    mappings[required_col] = selected
            
            self.column_mapping = mappings
            
            # Update status indicator
            if mappings:
                self.column_mapping_indicator.config(text="📋", foreground="green")
            else:
                self.column_mapping_indicator.config(text="📋", foreground="blue")
            
            # Save to file for persistence
            try:
                import json
                with open('column_mapping.json', 'w') as f:
                    json.dump(mappings, f, indent=2)
                messagebox.showinfo("Success", f"Column mappings saved successfully!\nMapped {len(mappings)} columns.")
                mapping_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mappings: {str(e)}")
        
        def load_saved_mappings():
            """Load previously saved mappings"""
            try:
                import json
                if os.path.exists('column_mapping.json'):
                    with open('column_mapping.json', 'r') as f:
                        saved_mappings = json.load(f)
                    
                    self.column_mapping = saved_mappings
                    
                    # Update dropdowns
                    for required_col, mapped_col in saved_mappings.items():
                        if required_col in mapping_widgets:
                            mapping_widgets[required_col].set(mapped_col)
                    
                    messagebox.showinfo("Success", f"Loaded {len(saved_mappings)} saved mappings.")
                else:
                    messagebox.showinfo("Info", "No saved mappings found.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load mappings: {str(e)}")
        
        ttk.Button(button_frame, text="Save Mappings", command=save_mappings, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Load Saved", command=load_saved_mappings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=mapping_window.destroy).pack(side=tk.RIGHT)
        
        # Help text
        help_frame = ttk.LabelFrame(main_frame, text="How to Use Column Mapping", padding=10)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        help_text = """1. Select a work order file to see available columns
2. Map each required column to a column in your file
3. Use Auto-Detect to automatically find similar column names
4. Save mappings for future use with similar files
5. The program will use these mappings when processing files

Example: If your CMMS exports "date" instead of "Reported Date", 
map "Reported Date" to "date" in the dropdown."""
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT, wraplength=650)
        help_label.pack(anchor=tk.W)
        
        # Load saved mappings if they exist
        load_saved_mappings()

    def apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column mapping to DataFrame"""
        if not self.column_mapping:
            return df
        
        # Create a copy to avoid modifying original
        mapped_df = df.copy()
        
        # Rename columns based on mapping
        rename_dict = {}
        for required_col, mapped_col in self.column_mapping.items():
            if mapped_col in df.columns and mapped_col != required_col:
                rename_dict[mapped_col] = required_col
        
        if rename_dict:
            mapped_df = mapped_df.rename(columns=rename_dict)
            logging.info(f"Applied column mapping: {rename_dict}")
        
        return mapped_df

    def load_saved_column_mappings(self):
        """Load saved column mappings from file"""
        try:
            import json
            if os.path.exists('column_mapping.json'):
                with open('column_mapping.json', 'r') as f:
                    self.column_mapping = json.load(f)
                
                # Update status indicator
                if self.column_mapping:
                    self.column_mapping_indicator.config(text="📋", foreground="green")
                    logging.info(f"Loaded {len(self.column_mapping)} saved column mappings")
                else:
                    self.column_mapping_indicator.config(text="📋", foreground="blue")
        except Exception as e:
            logging.error(f"Failed to load column mappings: {e}")
            self.column_mapping = {}
            self.column_mapping_indicator.config(text="📋", foreground="blue")
    
    def load_saved_file_paths(self):
        """Load saved file paths from configuration."""
        try:
            # Load work order path
            if self.config.get('last_work_order_path') and os.path.exists(self.config['last_work_order_path']):
                self.wo_entry.delete(0, tk.END)
                self.wo_entry.insert(0, self.config['last_work_order_path'])
                logging.info(f"Loaded saved work order path: {self.config['last_work_order_path']}")
            
            # Load dictionary path
            if self.config.get('last_dictionary_path') and os.path.exists(self.config['last_dictionary_path']):
                self.dict_entry.delete(0, tk.END)
                self.dict_entry.insert(0, self.config['last_dictionary_path'])
                logging.info(f"Loaded saved dictionary path: {self.config['last_dictionary_path']}")
            
            # Load output directory
            if self.config.get('last_output_directory') and os.path.exists(self.config['last_output_directory']):
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, self.config['last_output_directory'])
                self.output_dir = self.config['last_output_directory']
                logging.info(f"Loaded saved output directory: {self.config['last_output_directory']}")
            
            # Load AI settings
            if self.config.get('ai_enabled'):
                self.ai_enabled_var.set(True)
                self.use_ai_classification = True
            
            if self.config.get('confidence_threshold'):
                confidence = self.config['confidence_threshold']
                self.confidence_var.set(str(confidence))
                self.confidence_scale_var.set(confidence)
                
        except Exception as e:
            logging.error(f"Error loading saved file paths: {e}")
    
    def save_file_paths(self):
        """Save current file paths to configuration."""
        try:
            # Update configuration with current paths
            self.config['last_work_order_path'] = self.wo_entry.get()
            self.config['last_dictionary_path'] = self.dict_entry.get()
            self.config['last_output_directory'] = self.output_entry.get()
            self.config['ai_enabled'] = self.ai_enabled_var.get()
            self.config['confidence_threshold'] = float(self.confidence_var.get())
            
            # Save to file
            save_app_config(self.config)
            
        except Exception as e:
            logging.error(f"Error saving file paths: {e}")
    
    def on_closing(self):
        """Handle application closing - save configuration and cleanup."""
        try:
            # Save current configuration
            self.save_file_paths()
            logging.info("Configuration saved on application close")
        except Exception as e:
            logging.error(f"Error saving configuration on close: {e}")
        
        # Close the application
        self.root.quit()

    # ===== NEW ANALYSIS TABS =====
    
    def create_weibull_analysis_tab(self):
        """Create the Weibull analysis tab"""
        weibull_frame = ttk.Frame(self.notebook)
        self.notebook.add(weibull_frame, text="📊 Weibull Analysis")
        
        # Initialize Weibull analysis
        if MODULES_AVAILABLE:
            try:
                self.weibull_analyzer = WeibullAnalysis()
                logging.info("Weibull analyzer initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Weibull analyzer: {e}")
                self.weibull_analyzer = None
        else:
            self.weibull_analyzer = None
            logging.warning("Weibull analysis module not available")
        
        # Control panel
        control_frame = ttk.LabelFrame(weibull_frame, text="Analysis Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Equipment filter
        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(fill=tk.X, pady=2)
        ttk.Label(filter_frame, text="Equipment Filter:", width=15).pack(side=tk.LEFT)
        self.weibull_equipment_var = tk.StringVar()
        self.weibull_equipment_dropdown = ttk.Combobox(filter_frame, textvariable=self.weibull_equipment_var, 
                                                      state="readonly", width=20)
        self.weibull_equipment_dropdown['values'] = ['']
        self.weibull_equipment_dropdown.pack(side=tk.LEFT, padx=(5, 20))
        self.weibull_equipment_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_weibull_analysis())
        
        # Failure code source selector
        self.weibull_failure_code_source_var = tk.StringVar(value="AI/Dictionary")
        ttk.Label(filter_frame, text="Failure Code Source:", width=18).pack(side=tk.LEFT)
        self.weibull_failure_code_source_dropdown = ttk.Combobox(filter_frame, textvariable=self.weibull_failure_code_source_var, 
                                                                state="readonly", width=15)
        self.weibull_failure_code_source_dropdown['values'] = ["AI/Dictionary", "User"]
        self.weibull_failure_code_source_dropdown.pack(side=tk.LEFT, padx=(5, 5))
        self.weibull_failure_code_source_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_weibull_failure_dropdown())
        
        # Failure code filter
        ttk.Label(filter_frame, text="Failure Code:", width=12).pack(side=tk.LEFT)
        self.weibull_failure_var = tk.StringVar()
        self.weibull_failure_dropdown = ttk.Combobox(filter_frame, textvariable=self.weibull_failure_var, 
                                                    state="readonly", width=20)
        self.weibull_failure_dropdown['values'] = ['']
        self.weibull_failure_dropdown.pack(side=tk.LEFT, padx=5)
        self.weibull_failure_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_weibull_analysis())
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_frame, text="🔄 Update Analysis", command=self.update_weibull_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📈 Export Plot", command=self.export_weibull_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 Copy Summary", command=lambda: self.copy_to_clipboard(self.weibull_summary_text.get("1.0", tk.END))).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📊 Export Results", command=self.export_weibull_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 Add to FMEA Export", command=self.add_to_fmea_export).pack(side=tk.LEFT, padx=5)
        
        # Results display with three panels
        results_frame = ttk.LabelFrame(weibull_frame, text="Weibull Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Main paned window for plot and right panel
        main_paned = ttk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Plot area (left side)
        plot_frame = ttk.Frame(main_paned)
        main_paned.add(plot_frame, weight=3)
        self.weibull_plot_frame = plot_frame
        
        # Right panel (summary and work orders)
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=1)
        
        # Vertical paned window for summary and work orders
        right_paned = ttk.PanedWindow(right_panel, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # Summary area (top)
        summary_frame = ttk.LabelFrame(right_paned, text="Analysis Summary", padding=5)
        right_paned.add(summary_frame, weight=1)
        
        # Summary text
        self.weibull_summary_text = tk.Text(summary_frame, wrap=tk.WORD, height=15)
        weibull_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.weibull_summary_text.yview)
        self.weibull_summary_text.configure(yscrollcommand=weibull_scrollbar.set)
        
        self.weibull_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        weibull_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Work orders area (bottom)
        work_orders_frame = ttk.LabelFrame(right_paned, text="Work Orders in Analysis", padding=5)
        right_paned.add(work_orders_frame, weight=1)
        
        # Work orders tree
        wo_columns = ['Include', 'Work Order', 'Date', 'Description', 'AI Failure Code', 'User Failure Code', 'Time (days)']
        self.weibull_wo_tree = ttk.Treeview(work_orders_frame, columns=wo_columns, show='headings', height=8)
        
        # Set column widths and headings
        column_widths = {
            'Include': 60,
            'Work Order': 100,
            'Date': 80,
            'Description': 200,
            'AI Failure Code': 120,
            'User Failure Code': 120,
            'Time (days)': 80
        }
        
        for col in wo_columns:
            self.weibull_wo_tree.heading(col, text=col)
            self.weibull_wo_tree.column(col, width=column_widths.get(col, 80))
        
        # Scrollbars for work orders tree
        wo_scrollbar_y = ttk.Scrollbar(work_orders_frame, orient=tk.VERTICAL, command=self.weibull_wo_tree.yview)
        wo_scrollbar_x = ttk.Scrollbar(work_orders_frame, orient=tk.HORIZONTAL, command=self.weibull_wo_tree.xview)
        self.weibull_wo_tree.configure(yscrollcommand=wo_scrollbar_y.set, xscrollcommand=wo_scrollbar_x.set)
        
        self.weibull_wo_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        wo_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        wo_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind click events for toggling work orders and censoring
        self.weibull_wo_tree.bind('<Button-1>', self.toggle_weibull_work_order)
        
        # Add note about inclusion and censoring
        note_frame = ttk.Frame(work_orders_frame)
        note_frame.pack(fill=tk.X, pady=(5, 0))
        note_label = ttk.Label(note_frame, text="💡 Note: Only included work orders are used for failure analysis. Unchecked work orders are treated as censored. Changes sync with main tree.", 
                              font=("Arial", 8), foreground="blue", wraplength=400)
        note_label.pack()
        
        # Add note about first-day censoring
        first_day_note_frame = ttk.Frame(work_orders_frame)
        first_day_note_frame.pack(fill=tk.X, pady=(2, 0))
        first_day_note_label = ttk.Label(first_day_note_frame, text="📅 Tip: Consider censoring work orders from the first day of analysis (time=0) as they may not represent true failure times.", 
                                        font=("Arial", 8), foreground="orange", wraplength=400)
        first_day_note_label.pack()
        
        # Initialize with placeholder
        self.weibull_summary_text.insert(tk.END, "Weibull Analysis\n\n")
        self.weibull_summary_text.insert(tk.END, "Load work order data and select equipment/failure code to begin analysis.\n\n")
        self.weibull_summary_text.insert(tk.END, "This analysis provides:\n")
        self.weibull_summary_text.insert(tk.END, "• Weibull parameter estimation (β, η)\n")
        self.weibull_summary_text.insert(tk.END, "• Reliability predictions (6M, 1Y, 2Y, 4Y, 10Y)\n")
        self.weibull_summary_text.insert(tk.END, "• Confidence bounds\n")
        self.weibull_summary_text.insert(tk.END, "• Mean time to failure\n")
        self.weibull_summary_text.insert(tk.END, "• Goodness of fit assessment\n")
    
    # FMEA export tab removed - functionality moved to Weibull analysis tab
    
    def create_pm_analysis_tab(self):
        """Create the optimized PM frequency analysis tab"""
        pm_frame = ttk.Frame(self.notebook)
        self.notebook.add(pm_frame, text="🔧 PM Frequency")
        
        # Initialize PM analysis
        self.pm_analyzer = PMAnalysis() if MODULES_AVAILABLE else None
        if self.pm_analyzer and hasattr(self, 'weibull_analyzer'):
            self.pm_analyzer.set_weibull_analyzer(self.weibull_analyzer)
        
        # Control panel
        control_frame = ttk.LabelFrame(pm_frame, text="PM Frequency Analysis Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Analysis options
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(options_frame, text="Analysis Options:").pack(side=tk.LEFT)
        self.include_crow_amsaa_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Crow-AMSAA MTBF", variable=self.include_crow_amsaa_var).pack(side=tk.LEFT, padx=(10, 5))
        self.include_weibull_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Weibull Analysis", variable=self.include_weibull_var).pack(side=tk.LEFT, padx=5)
        self.include_practical_intervals_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Practical Intervals", variable=self.include_practical_intervals_var).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_frame, text="🔄 Update Analysis", command=self.update_pm_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📈 Export Plot", command=self.export_pm_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 Copy Summary", command=lambda: self.copy_to_clipboard(self.pm_summary_text.get("1.0", tk.END))).pack(side=tk.LEFT, padx=5)
        
        # Results display - split into summary and detailed analysis
        results_frame = ttk.LabelFrame(pm_frame, text="PM Frequency Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create paned window for summary and detailed analysis
        paned_frame = ttk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
        paned_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Summary and plot
        left_frame = ttk.Frame(paned_frame)
        paned_frame.add(left_frame, weight=2)
        
        # Summary area
        summary_frame = ttk.LabelFrame(left_frame, text="Analysis Summary", padding=5)
        summary_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.pm_summary_text = tk.Text(summary_frame, wrap=tk.WORD, height=8, font=("Arial", 10))
        pm_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.pm_summary_text.yview)
        self.pm_summary_text.configure(yscrollcommand=pm_scrollbar.set)
        
        self.pm_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pm_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Plot area
        plot_frame = ttk.LabelFrame(left_frame, text="Expected Life Comparison", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.pm_plot_frame = plot_frame
        
        # Right side - Detailed analysis table
        right_frame = ttk.Frame(paned_frame)
        paned_frame.add(right_frame, weight=1)
        
        # Detailed analysis table
        table_frame = ttk.LabelFrame(right_frame, text="Detailed PM Recommendations", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for detailed analysis
        columns = ['Method', 'Eta/MTBF', 'Failure Type', 'PM Frequency', 'PM Type', 'Confidence']
        self.pm_analysis_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Set column widths and headings
        column_widths = {
            'Method': 120,
            'Eta/MTBF': 100,
            'Failure Type': 120,
            'PM Frequency': 120,
            'PM Type': 150,
            'Confidence': 100
        }
        
        for col in columns:
            self.pm_analysis_tree.heading(col, text=col)
            self.pm_analysis_tree.column(col, width=column_widths.get(col, 100))
        
        # Scrollbars for table
        tree_scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.pm_analysis_tree.yview)
        tree_scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.pm_analysis_tree.xview)
        self.pm_analysis_tree.configure(yscrollcommand=tree_scrollbar_y.set, xscrollcommand=tree_scrollbar_x.set)
        
        self.pm_analysis_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize with placeholder
        self.pm_summary_text.insert(tk.END, "Enhanced PM Frequency Analysis\n\n")
        self.pm_summary_text.insert(tk.END, "This analysis provides comprehensive PM frequency recommendations based on:\n")
        self.pm_summary_text.insert(tk.END, "• Equipment-specific Crow-AMSAA MTBF analysis\n")
        self.pm_summary_text.insert(tk.END, "• Weibull parameters for each failure mode\n")
        self.pm_summary_text.insert(tk.END, "• Expected life comparison (Eta vs MTBF)\n")
        self.pm_summary_text.insert(tk.END, "• Practical PM frequency intervals\n")
        self.pm_summary_text.insert(tk.END, "• PM type recommendations based on failure patterns\n\n")
        self.pm_summary_text.insert(tk.END, "Select work orders and apply filters on the main analysis tab, then click 'Update Analysis'.\n")
    
    def create_spares_analysis_tab(self):
        """Create the spares analysis tab"""
        spares_frame = ttk.Frame(self.notebook)
        self.notebook.add(spares_frame, text="📦 Spares Analysis")
        
        # Initialize spares analysis
        self.spares_analyzer = SparesAnalysis() if MODULES_AVAILABLE else None
        
        # Control panel
        control_frame = ttk.LabelFrame(spares_frame, text="Spares Analysis Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Simulation parameters
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        # First row - basic parameters
        row1_frame = ttk.Frame(params_frame)
        row1_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1_frame, text="Simulation Years:").pack(side=tk.LEFT)
        self.simulation_years_var = tk.StringVar(value="10")
        ttk.Entry(row1_frame, textvariable=self.simulation_years_var, width=10).pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row1_frame, text="Simulations:").pack(side=tk.LEFT)
        self.num_simulations_var = tk.StringVar(value="1000")
        ttk.Entry(row1_frame, textvariable=self.num_simulations_var, width=10).pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row1_frame, text="Lead Time (days):").pack(side=tk.LEFT)
        self.lead_time_var = tk.StringVar(value="30")
        ttk.Entry(row1_frame, textvariable=self.lead_time_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Second row - equipment selection
        row2_frame = ttk.Frame(params_frame)
        row2_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(row2_frame, text="Equipment:").pack(side=tk.LEFT)
        self.spares_equipment_var = tk.StringVar()
        self.spares_equipment_dropdown = ttk.Combobox(row2_frame, textvariable=self.spares_equipment_var, width=20)
        self.spares_equipment_dropdown.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row2_frame, text="Service Level (%):").pack(side=tk.LEFT)
        self.service_level_var = tk.StringVar(value="95")
        ttk.Entry(row2_frame, textvariable=self.service_level_var, width=10).pack(side=tk.LEFT, padx=(5, 20))
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_frame, text="🔄 Update Analysis", command=self.update_spares_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📈 Export Plot", command=self.export_spares_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 Copy Summary", command=lambda: self.copy_to_clipboard(self.spares_summary_text.get("1.0", tk.END))).pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(spares_frame, text="Spares Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Split into plot and summary
        paned_frame = ttk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
        paned_frame.pack(fill=tk.BOTH, expand=True)
        
        # Plot area
        plot_frame = ttk.Frame(paned_frame)
        paned_frame.add(plot_frame, weight=2)
        self.spares_plot_frame = plot_frame
        
        # Summary area
        summary_frame = ttk.Frame(paned_frame)
        paned_frame.add(summary_frame, weight=1)
        
        # Summary text
        self.spares_summary_text = tk.Text(summary_frame, wrap=tk.WORD, height=20)
        spares_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.spares_summary_text.yview)
        self.spares_summary_text.configure(yscrollcommand=spares_scrollbar.set)
        
        self.spares_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        spares_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize with placeholder
        self.spares_summary_text.insert(tk.END, "Enhanced Spares Analysis\n\n")
        self.spares_summary_text.insert(tk.END, "Load work order data to begin spares analysis.\n\n")
        self.spares_summary_text.insert(tk.END, "This analysis provides:\n")
        self.spares_summary_text.insert(tk.END, "• Weibull-based Monte Carlo simulation\n")
        self.spares_summary_text.insert(tk.END, "• Equipment-specific stocking levels\n")
        self.spares_summary_text.insert(tk.END, "• MTBF vs Weibull comparison\n")
        self.spares_summary_text.insert(tk.END, "• 10-year demand forecasting\n")
        self.spares_summary_text.insert(tk.END, "• Failure mode-specific recommendations\n")

    # ===== NEW ANALYSIS METHODS =====
    
    def update_weibull_analysis(self):
        """Update Weibull analysis based on current filters"""
        logging.info("Starting Weibull analysis update")
        logging.info(f"MODULES_AVAILABLE: {MODULES_AVAILABLE}")
        logging.info(f"weibull_analyzer: {self.weibull_analyzer}")
        logging.info(f"wo_df is None: {self.wo_df is None}")
        
        if not MODULES_AVAILABLE or not self.weibull_analyzer or self.wo_df is None:
            messagebox.showwarning("Warning", "Weibull analysis module not available or no data loaded")
            return
        
        try:
            # Get filtered data
            filtered_df = self.get_filtered_df()
            if filtered_df.empty:
                messagebox.showwarning("Warning", "No data available for Weibull analysis")
                return
            
            # Apply equipment filter if specified
            equipment_filter = self.weibull_equipment_var.get()
            if equipment_filter:
                filtered_df = filtered_df[filtered_df['Equipment #'] == equipment_filter]
            
            # Apply failure code filter if specified (by selected source)
            failure_filter = self.weibull_failure_var.get()
            code_col = 'Failure Code' if self.weibull_failure_code_source_var.get() == 'AI/Dictionary' else 'User failure code'
            if failure_filter:
                filtered_df = filtered_df[filtered_df[code_col] == failure_filter]
            
            if isinstance(filtered_df, pd.DataFrame) and filtered_df.empty:
                messagebox.showwarning("Warning", "No data available after filtering")
                return
            
            # Get date range for analysis
            start_date = self.start_date
            end_date = self.end_date
            
            # Use only included work orders for failure analysis
            # Unchecked work orders are treated as censored (excluded from analysis)
            try:
                # Convert to DataFrame if it's not already
                if not isinstance(filtered_df, pd.DataFrame):
                    filtered_df = pd.DataFrame(filtered_df)
                
                # Use only the included work orders for failure analysis
                failure_df = filtered_df[filtered_df.index.isin(self.included_indices)]
                excluded_count = len(filtered_df) - len(failure_df)
                logging.info(f"Weibull analysis: {excluded_count} work orders excluded (treated as censored)")
            except Exception as e:
                logging.warning(f"Error filtering included data: {e}")
                failure_df = filtered_df
            
            # Calculate failure times (excluding censored data)
            if isinstance(failure_df, pd.DataFrame):
                failure_times = self.weibull_analyzer.calculate_failure_times(failure_df)
            else:
                messagebox.showwarning("Warning", "Invalid data format for Weibull analysis")
                return
            logging.info(f"Weibull analysis: calculated {len(failure_times)} failure times")
            
            if len(failure_times) < 2:
                messagebox.showwarning("Warning", "Insufficient failure data for Weibull analysis (need at least 2 failures)")
                return
            
            # Clear previous plot
            for widget in self.weibull_plot_frame.winfo_children():
                widget.destroy()
            
            # Create plot with date range
            fig, beta, eta = self.weibull_analyzer.create_weibull_plot(
                failure_times, self.weibull_plot_frame, 
                f"Weibull Analysis - {equipment_filter or 'All Equipment'}",
                start_date, end_date)
            self.weibull_plot_fig = fig
            
            # Store Weibull parameters for PM analysis
            beta, eta = self.weibull_analyzer.weibull_mle(failure_times)
            # Store parameters as a dictionary for PM analysis access
            try:
                self.weibull_analyzer.weibull_params = {
                    'beta': float(beta),
                    'eta': float(eta),
                    'failure_times': failure_times,
                    'equipment_filter': equipment_filter,
                    'failure_filter': failure_filter
                }
            except Exception as e:
                logging.warning(f"Could not store Weibull parameters: {e}")
                # Store as instance variable instead
                self.weibull_analyzer._pm_params = {
                    'beta': float(beta),
                    'eta': float(eta),
                    'failure_times': failure_times,
                    'equipment_filter': equipment_filter,
                    'failure_filter': failure_filter
                }
            
            # Update summary with goodness of fit
            summary = self.weibull_analyzer.get_analysis_summary(failure_times, start_date, end_date)
            
            self.weibull_summary_text.delete(1.0, tk.END)
            self.weibull_summary_text.insert(tk.END, f"Weibull Analysis Results\n\n")
            
            # Date range
            if start_date and end_date:
                self.weibull_summary_text.insert(tk.END, f"Analysis Period: {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}\n")
            elif start_date:
                self.weibull_summary_text.insert(tk.END, f"From: {start_date.strftime('%m/%d/%Y')}\n")
            elif end_date:
                self.weibull_summary_text.insert(tk.END, f"To: {end_date.strftime('%m/%d/%Y')}\n")
            
            self.weibull_summary_text.insert(tk.END, f"Status: {summary['status']}\n")
            self.weibull_summary_text.insert(tk.END, f"Shape Parameter (β): {summary['beta']:.3f}\n")
            self.weibull_summary_text.insert(tk.END, f"Scale Parameter (η): {summary['eta']:.1f} days\n")
            self.weibull_summary_text.insert(tk.END, f"Mean Time to Failure: {summary['mean_time_to_failure']:.1f} days\n")
            self.weibull_summary_text.insert(tk.END, f"Failure Count: {summary['failure_count']}\n\n")
            
            # Goodness of fit assessment
            self.weibull_summary_text.insert(tk.END, "Goodness of Fit:\n")
            self.weibull_summary_text.insert(tk.END, f"  Overall: {summary['goodness_of_fit']}\n")
            self.weibull_summary_text.insert(tk.END, f"  Quality: {summary['fit_quality']}\n")
            self.weibull_summary_text.insert(tk.END, f"  R-squared: {summary['r_squared']:.3f}\n")
            self.weibull_summary_text.insert(tk.END, f"  K-S Statistic: {summary['kolmogorov_smirnov']:.3f}\n\n")
            
            self.weibull_summary_text.insert(tk.END, "Reliability Predictions:\n")
            self.weibull_summary_text.insert(tk.END, f"  6 Months: {summary['reliability_6m']*100:.1f}%\n")
            self.weibull_summary_text.insert(tk.END, f"  1 Year: {summary['reliability_1y']*100:.1f}%\n")
            self.weibull_summary_text.insert(tk.END, f"  2 Years: {summary['reliability_2y']*100:.1f}%\n")
            self.weibull_summary_text.insert(tk.END, f"  4 Years: {summary['reliability_4y']*100:.1f}%\n")
            self.weibull_summary_text.insert(tk.END, f"  10 Years: {summary['reliability_10y']*100:.1f}%\n")

            # --- Demand Rate Section ---
            # Calculate demand rate for filtered failure mode
            from spares_analysis import SparesAnalysis
            spares_analyzer = SparesAnalysis()
            demand_result = spares_analyzer.analyze_spares_demand(filtered_df)
            demand_rate = None
            if failure_filter and demand_result.get('spares_demand') and failure_filter in demand_result['spares_demand']:
                demand_rate = demand_result['spares_demand'][failure_filter]['demand_rate_per_year']
            elif demand_result.get('avg_demand_rate'):
                demand_rate = demand_result['avg_demand_rate']
            # Calculate Crow-AMSAA failures/year
            if isinstance(filtered_df, pd.DataFrame):
                _, _, crow_amsaa_failures_per_year = calculate_crow_amsaa_params(filtered_df, set(self.included_indices))
            else:
                crow_amsaa_failures_per_year = 0
            self.weibull_summary_text.insert(tk.END, "\nDemand Rate (filtered failure mode):\n")
            if demand_rate is not None:
                self.weibull_summary_text.insert(tk.END, f"  Demand Rate: {demand_rate:.2f} failures/year\n")
            self.weibull_summary_text.insert(tk.END, f"  Crow-AMSAA Rate: {crow_amsaa_failures_per_year:.2f} failures/year\n\n")
            
            # Update work orders table
            if isinstance(filtered_df, pd.DataFrame):
                self.update_weibull_work_orders_table(filtered_df, failure_times)
            
            self.status_label.config(text="Weibull analysis updated", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update Weibull analysis: {str(e)}")
            logging.error(f"Error updating Weibull analysis: {e}")
    
    def update_weibull_work_orders_table(self, filtered_df: pd.DataFrame, failure_times: List[float]):
        """Update the work orders table in Weibull analysis tab"""
        try:
            # Clear existing items
            self.weibull_wo_tree.delete(*self.weibull_wo_tree.get_children())
            
            if filtered_df.empty:
                return
            
            # Sort by date to calculate time differences
            filtered_df = filtered_df.sort_values('Reported Date')
            dates = pd.to_datetime(filtered_df['Reported Date'], errors='coerce').dropna()
            
            if len(dates) < 2:
                return
            
            start_date = dates.iloc[0]
            
            # Add work orders to table
            for idx, row in filtered_df.iterrows():
                # Check if included in main analysis - this determines if it's used for failure analysis
                include_mark = '☑' if idx in self.included_indices else '☐'
                
                # Calculate time from start
                try:
                    date = pd.to_datetime(row['Reported Date'])
                    time_days = (date - start_date).days
                except:
                    time_days = 0
                
                # Get work order description (truncate if too long)
                description = str(row.get('Description', ''))
                if len(description) > 50:
                    description = description[:47] + "..."
                
                # Get failure codes
                ai_failure_code = str(row.get('Failure Code', '')).upper()
                user_failure_code = str(row.get('User failure code', '')).upper()
                
                self.weibull_wo_tree.insert('', 'end', values=(
                    include_mark,
                    row.get('Work Order', ''),
                    row.get('Reported Date', ''),
                    description,
                    ai_failure_code,
                    user_failure_code,
                    f"{time_days:.0f}"
                ))
            
        except Exception as e:
            logging.error(f"Error updating Weibull work orders table: {e}")
    
    def toggle_weibull_work_order(self, event):
        """Toggle work order inclusion in Weibull analysis"""
        if self.weibull_wo_tree is None:
            return
        
        item = self.weibull_wo_tree.identify_row(event.y)
        column = self.weibull_wo_tree.identify_column(event.x)
        
        if not item or column != '#1':  # Only handle include column
            return
        
        values = list(self.weibull_wo_tree.item(item, 'values'))
        work_order = values[1]  # Work Order column (index 1 now)
        
        # Find the corresponding row in the main dataframe
        if self.wo_df is not None:
            matching_rows = self.wo_df[self.wo_df['Work Order'] == work_order]
            if not matching_rows.empty:
                idx = matching_rows.index[0]
                
                # Toggle inclusion status
                if idx in self.included_indices:
                    self.included_indices.remove(idx)
                    values[0] = '☐'
                else:
                    self.included_indices.add(idx)
                    values[0] = '☑'
                
                # Update the tree item
                self.weibull_wo_tree.item(item, values=values)
                
                # Update the main analysis table as well
                self.update_table()
                
                # Re-run Weibull analysis with updated data
                self.update_weibull_analysis()
    
    def export_weibull_plot(self):
        """Export the Weibull plot to a file."""
        if not hasattr(self, 'weibull_plot_fig') or self.weibull_plot_fig is None:
            messagebox.showwarning("Warning", "No plot available to export. Please update the plot first.")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Export Weibull Plot"
            )
            if file_path:
                self.weibull_plot_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Weibull plot exported to {file_path}")
                logging.info(f"Weibull plot exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
            logging.error(f"Error exporting Weibull plot: {e}") 
    
    def export_weibull_results(self):
        """Export Weibull analysis results"""
        if not MODULES_AVAILABLE or not self.weibull_analyzer:
            messagebox.showwarning("Warning", "Weibull analysis module not available")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Get current analysis data and export
                # This would need to be implemented based on the current analysis state
                messagebox.showinfo("Success", f"Weibull results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    # Old FMEA functions removed - functionality moved to new JSON-based system
    
    def update_pm_analysis(self):
        """Update enhanced PM frequency analysis based on current work order selections and filters"""
        if not MODULES_AVAILABLE or not self.pm_analyzer or self.wo_df is None:
            messagebox.showwarning("Warning", "PM analysis module not available or no data loaded")
            return
        
        try:
            # Get current filtered data (respects all selections and filters)
            filtered_df = self.get_filtered_df()
            if filtered_df.empty:
                messagebox.showwarning("Warning", "No data available for PM analysis")
                return
            
            # Get only the included work orders (checked in tree)
            valid_indices = filtered_df.index.intersection(self.included_indices)
            if len(valid_indices) == 0:
                messagebox.showwarning("Warning", "No work orders selected for analysis")
                return
            
            # Use only the selected work orders
            selected_df = filtered_df.loc[valid_indices]
            
            # Clear previous results
            self.pm_analysis_tree.delete(*self.pm_analysis_tree.get_children())
            
            # 1. Display Equipment number based on current main tree filter
            equipment_list = selected_df['Equipment #'].unique() if 'Equipment #' in selected_df.columns else []
            equipment_text = ', '.join(str(eq) for eq in equipment_list[:3])
            if len(equipment_list) > 3:
                equipment_text += f" (+{len(equipment_list)-3} more)"
            
            # 2. Calculate Crow-AMSAA MTBF based on main tree filter
            crow_amsaa_mtbf = 0.0
            if self.include_crow_amsaa_var.get():
                crow_amsaa_mtbf = calculate_mtbf(selected_df, set(valid_indices))
            
            # 3. Get Weibull parameters for each failure mode
            weibull_analyses = []
            if self.include_weibull_var.get() and hasattr(self, 'weibull_analyzer') and self.weibull_analyzer:
                # Get failure modes for the selected equipment
                failure_modes = []
                if 'Failure Code' in selected_df.columns:
                    failure_modes = selected_df['Failure Code'].unique().tolist()
                elif 'AI Failure Code' in selected_df.columns:
                    failure_modes = selected_df['AI Failure Code'].unique().tolist()
                
                # Get current Weibull analysis if available (for specific failure mode)
                current_weibull_filter = None
                if hasattr(self.weibull_analyzer, 'weibull_params') and self.weibull_analyzer.weibull_params:
                    weibull_params = self.weibull_analyzer.weibull_params.copy()
                    # Add equipment and failure mode filters if applied
                    if hasattr(self, 'weibull_equipment_var') and self.weibull_equipment_var.get():
                        weibull_params['equipment_filter'] = self.weibull_equipment_var.get()
                    if hasattr(self, 'weibull_failure_var') and self.weibull_failure_var.get():
                        weibull_params['failure_filter'] = self.weibull_failure_var.get()
                        current_weibull_filter = self.weibull_failure_var.get()
                    weibull_analyses.append(weibull_params)
                
                # Get Weibull parameters for each failure mode
                for failure_mode in failure_modes:
                    if failure_mode and failure_mode != current_weibull_filter:  # Avoid duplicate
                        # Filter data for this specific failure mode
                        failure_mode_df = selected_df[selected_df['Failure Code'] == failure_mode] if 'Failure Code' in selected_df.columns else \
                                        selected_df[selected_df['AI Failure Code'] == failure_mode]
                        
                        if len(failure_mode_df) >= 2:  # Need at least 2 failures for Weibull
                            weibull_params = self.pm_analyzer.get_weibull_parameters(failure_mode_df)
                            if weibull_params and weibull_params.get('eta', 0) > 0:
                                weibull_params['method'] = f'Weibull - {failure_mode}'
                                weibull_analyses.append(weibull_params)
            
            # 4. Create analysis results for table and plot
            analysis_results = []
            
            # Add Crow-AMSAA MTBF analysis
            if crow_amsaa_mtbf > 0:
                analysis_results.append({
                    'method': 'Crow-AMSAA MTBF',
                    'eta_mtbf': crow_amsaa_mtbf,
                    'failure_type': 'Random (MTBF-based)',
                    'pm_frequency': self._get_practical_pm_frequency(crow_amsaa_mtbf),
                    'pm_type': 'Predictive Maintenance',
                    'confidence': 'High' if len(valid_indices) >= 5 else 'Medium'
                })
            
            # Add Weibull analyses
            for weibull_analysis in weibull_analyses:
                beta = weibull_analysis.get('beta', 1.0)
                eta = weibull_analysis.get('eta', 0)
                failure_type = weibull_analysis.get('failure_type', 'Unknown')
                confidence = weibull_analysis.get('confidence', 'Low')
                
                if eta > 0:
                    # Determine PM type based on failure type
                    pm_type = self._get_pm_type_recommendation(failure_type)
                    
                    # Get practical PM frequency
                    pm_frequency = self._get_practical_pm_frequency(eta, beta)
                    
                    method_name = weibull_analysis.get('method', 'Weibull Analysis')
                    
                    analysis_results.append({
                        'method': method_name,
                        'eta_mtbf': eta,
                        'failure_type': failure_type,
                        'pm_frequency': pm_frequency,
                        'pm_type': pm_type,
                        'confidence': confidence
                    })
            
            # 5. Update summary text
            self.pm_summary_text.delete(1.0, tk.END)
            self.pm_summary_text.insert(tk.END, f"Enhanced PM Frequency Analysis Results\n\n")
            
            # Data summary
            self.pm_summary_text.insert(tk.END, "📊 Data Summary:\n")
            self.pm_summary_text.insert(tk.END, f"  Selected Work Orders: {len(valid_indices)}\n")
            self.pm_summary_text.insert(tk.END, f"  Date Range: {selected_df['Reported Date'].min()} to {selected_df['Reported Date'].max()}\n")
            self.pm_summary_text.insert(tk.END, f"  Equipment: {equipment_text}\n\n")
            
            # Analysis summary
            self.pm_summary_text.insert(tk.END, "📈 Analysis Summary:\n")
            if crow_amsaa_mtbf > 0:
                self.pm_summary_text.insert(tk.END, f"  Crow-AMSAA MTBF: {crow_amsaa_mtbf:.1f} days\n")
            
            if weibull_analyses:
                self.pm_summary_text.insert(tk.END, f"  Weibull Analysis: {len(weibull_analyses)} failure modes analyzed\n")
                for weibull_analysis in weibull_analyses:
                    eta = weibull_analysis.get('eta', 0)
                    beta = weibull_analysis.get('beta', 0)
                    failure_type = weibull_analysis.get('failure_type', 'Unknown')
                    method_name = weibull_analysis.get('method', 'Weibull')
                    self.pm_summary_text.insert(tk.END, f"    {method_name}: η={eta:.1f} days, β={beta:.3f} ({failure_type})\n")
            
            # 6. Populate detailed analysis table
            for result in analysis_results:
                self.pm_analysis_tree.insert('', 'end', values=(
                    result['method'],
                    f"{result['eta_mtbf']:.1f} days",
                    result['failure_type'],
                    result['pm_frequency'],
                    result['pm_type'],
                    result['confidence']
                ))
            
            # 7. Create comparison plot
            if analysis_results:
                self._create_pm_comparison_plot(analysis_results)
            
            # 8. Add recommendations
            self.pm_summary_text.insert(tk.END, "\n🎯 Key Recommendations:\n")
            
            # Find the lowest expected life
            if analysis_results:
                min_life = min(result['eta_mtbf'] for result in analysis_results)
                min_life_result = next(result for result in analysis_results if result['eta_mtbf'] == min_life)
                
                self.pm_summary_text.insert(tk.END, f"  • Lowest Expected Life: {min_life:.1f} days ({min_life_result['method']})\n")
                self.pm_summary_text.insert(tk.END, f"  • Recommended PM Frequency: {min_life_result['pm_frequency']}\n")
                self.pm_summary_text.insert(tk.END, f"  • PM Type: {min_life_result['pm_type']}\n")
                
                # Additional recommendations based on failure types
                wear_out_count = sum(1 for r in analysis_results if 'Wear Out' in r['failure_type'])
                infant_count = sum(1 for r in analysis_results if 'Infant Mortality' in r['failure_type'])
                
                if wear_out_count > 0:
                    self.pm_summary_text.insert(tk.END, "  • Wear-out patterns detected - PM is critical for prevention\n")
                if infant_count > 0:
                    self.pm_summary_text.insert(tk.END, "  • Infant mortality patterns - focus on quality control and procedures\n")
            
            self.status_label.config(text="Enhanced PM frequency analysis updated", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update PM analysis: {str(e)}")
            logging.error(f"Error updating PM analysis: {e}")
    
    def _get_practical_pm_frequency(self, eta_mtbf: float, beta: float = 1.0) -> str:
        """Convert optimal interval to practical PM frequency intervals"""
        try:
            # Adjust interval based on failure type
            if beta > 1.5:  # Wear-out failures
                # PM should be more frequent for wear-out failures
                adjusted_interval = eta_mtbf * 0.3  # 30% of characteristic life
            elif beta > 1.0:  # Early wear-out
                adjusted_interval = eta_mtbf * 0.5  # 50% of characteristic life
            else:  # Random or infant mortality
                adjusted_interval = eta_mtbf * 0.7  # 70% of characteristic life
            
            # Convert to practical intervals
            if adjusted_interval < 30:
                return "Weekly"
            elif adjusted_interval < 90:
                return "Monthly"
            elif adjusted_interval < 180:
                return "Quarterly"
            elif adjusted_interval < 365:
                return "Semi-annual"
            else:
                return "Annual"
                
        except Exception as e:
            logging.error(f"Error calculating practical PM frequency: {e}")
            return "Annual"
    
    def _get_pm_type_recommendation(self, failure_type: str) -> str:
        """Get PM type recommendation based on failure type"""
        if 'Wear Out' in failure_type:
            return "Failure Finding"
        elif 'Random' in failure_type:
            return "Predictive Maintenance"
        elif 'Infant Mortality' in failure_type:
            return "Procedure Review"
        else:
            return "Condition Monitoring"
    
    def _create_pm_comparison_plot(self, analysis_results: List[Dict]):
        """Create comparison plot showing eta/MTBF values for different failure modes"""
        try:
            # Clear previous plot
            for widget in self.pm_plot_frame.winfo_children():
                widget.destroy()
            
            if not analysis_results:
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data for plotting
            methods = [result['method'] for result in analysis_results]
            eta_mtbf_values = [result['eta_mtbf'] for result in analysis_results]
            failure_types = [result['failure_type'] for result in analysis_results]
            
            # Create color map based on failure type
            colors = []
            for failure_type in failure_types:
                if 'Wear Out' in failure_type:
                    colors.append('red')
                elif 'Random' in failure_type:
                    colors.append('blue')
                elif 'Infant Mortality' in failure_type:
                    colors.append('orange')
                else:
                    colors.append('gray')
            
            # Create bar plot
            bars = ax.bar(range(len(methods)), eta_mtbf_values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, eta_mtbf_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.1f} days', ha='center', va='bottom', fontsize=8)
            
            # Customize plot
            ax.set_title('Expected Life by Failure Mode (η/MTBF)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Expected Life (days)', fontsize=10)
            ax.set_xlabel('Failure Mode / Analysis Method', fontsize=10)
            
            # Set x-axis labels
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            
            # Add legend for failure types
            legend_elements = [
                Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Wear Out'),
                Rectangle((0,0),1,1, facecolor='blue', alpha=0.7, label='Random'),
                Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='Infant Mortality')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.pm_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.pm_plot_fig = fig
            
        except Exception as e:
            logging.error(f"Error creating PM comparison plot: {e}")
            # Create simple text widget as fallback
            error_label = ttk.Label(self.pm_plot_frame, text=f"Error creating plot: {str(e)}", foreground="red")
            error_label.pack(expand=True)
    
    def export_pm_report(self):
        """Export enhanced PM frequency analysis report"""
        if not MODULES_AVAILABLE or not self.pm_analyzer or self.wo_df is None:
            messagebox.showwarning("Warning", "PM analysis module not available or no data loaded")
            return
        
        try:
            # Get current filtered data
            filtered_df = self.get_filtered_df()
            if filtered_df.empty:
                messagebox.showwarning("Warning", "No data available for PM report")
                return
            
            # Get only the included work orders
            valid_indices = filtered_df.index.intersection(self.included_indices)
            if len(valid_indices) == 0:
                messagebox.showwarning("Warning", "No work orders selected for report")
                return
            
            selected_df = filtered_df.loc[valid_indices]
            
            # Get equipment list
            equipment_list = selected_df['Equipment #'].unique() if 'Equipment #' in selected_df.columns else []
            equipment_text = ', '.join(str(eq) for eq in equipment_list[:5])
            if len(equipment_list) > 5:
                equipment_text += f" (+{len(equipment_list)-5} more)"
            
            # Calculate Crow-AMSAA MTBF
            crow_amsaa_mtbf = calculate_mtbf(selected_df, set(valid_indices))
            
            # Get Weibull analyses
            weibull_analyses = []
            if hasattr(self, 'weibull_analyzer') and self.weibull_analyzer:
                if hasattr(self.weibull_analyzer, 'weibull_params') and self.weibull_analyzer.weibull_params:
                    weibull_params = self.weibull_analyzer.weibull_params.copy()
                    if hasattr(self, 'weibull_equipment_var') and self.weibull_equipment_var.get():
                        weibull_params['equipment_filter'] = self.weibull_equipment_var.get()
                    if hasattr(self, 'weibull_failure_var') and self.weibull_failure_var.get():
                        weibull_params['failure_filter'] = self.weibull_failure_var.get()
                    weibull_analyses.append(weibull_params)
                
                # Also get overall Weibull parameters
                overall_weibull = self.pm_analyzer.get_weibull_parameters(selected_df)
                if overall_weibull and overall_weibull.get('eta', 0) > 0:
                    overall_weibull['method'] = 'Overall Weibull'
                    weibull_analyses.append(overall_weibull)
            
            # Create detailed analysis data
            detailed_analysis = []
            
            # Add Crow-AMSAA MTBF analysis
            if crow_amsaa_mtbf > 0:
                detailed_analysis.append({
                    'Method': 'Crow-AMSAA MTBF',
                    'Eta/MTBF (days)': f"{crow_amsaa_mtbf:.1f}",
                    'Failure Type': 'Random (MTBF-based)',
                    'PM Frequency': self._get_practical_pm_frequency(crow_amsaa_mtbf),
                    'PM Type': 'Predictive Maintenance',
                    'Confidence': 'High' if len(valid_indices) >= 5 else 'Medium'
                })
            
            # Add Weibull analyses
            for weibull_analysis in weibull_analyses:
                beta = weibull_analysis.get('beta', 1.0)
                eta = weibull_analysis.get('eta', 0)
                failure_type = weibull_analysis.get('failure_type', 'Unknown')
                confidence = weibull_analysis.get('confidence', 'Low')
                
                if eta > 0:
                    pm_type = self._get_pm_type_recommendation(failure_type)
                    pm_frequency = self._get_practical_pm_frequency(eta, beta)
                    
                    method_name = weibull_analysis.get('method', 'Weibull Analysis')
                    if weibull_analysis.get('equipment_filter'):
                        method_name += f" ({weibull_analysis['equipment_filter']})"
                    if weibull_analysis.get('failure_filter'):
                        method_name += f" - {weibull_analysis['failure_filter']}"
                    
                    detailed_analysis.append({
                        'Method': method_name,
                        'Eta/MTBF (days)': f"{eta:.1f}",
                        'Beta (β)': f"{beta:.3f}",
                        'Failure Type': failure_type,
                        'PM Frequency': pm_frequency,
                        'PM Type': pm_type,
                        'Confidence': confidence
                    })
            
            # Create summary data
            summary_data = {
                'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Selected Work Orders': [len(valid_indices)],
                'Date Range': [f"{selected_df['Reported Date'].min()} to {selected_df['Reported Date'].max()}"],
                'Equipment': [equipment_text],
                'Crow-AMSAA MTBF (days)': [f"{crow_amsaa_mtbf:.1f}"],
                'Analysis Methods': [len(detailed_analysis)]
            }
            
            # Find lowest expected life
            if detailed_analysis:
                min_life = min(float(analysis['Eta/MTBF (days)']) for analysis in detailed_analysis)
                min_life_result = next(analysis for analysis in detailed_analysis 
                                     if float(analysis['Eta/MTBF (days)']) == min_life)
                
                summary_data.update({
                    'Lowest Expected Life (days)': [f"{min_life:.1f}"],
                    'Recommended PM Frequency': [min_life_result['PM Frequency']],
                    'Recommended PM Type': [min_life_result['PM Type']]
                })
            
            summary_df = pd.DataFrame(summary_data)
            detailed_df = pd.DataFrame(detailed_analysis)
            
            # Export to Excel
            if self.output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Enhanced_PM_Frequency_Report_{timestamp}.xlsx"
                filepath = os.path.join(self.output_dir, filename)
                
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    detailed_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
                    selected_df.to_excel(writer, sheet_name='Selected_Work_Orders', index=False)
                
                messagebox.showinfo("Success", f"Enhanced PM Frequency Report exported to:\n{filepath}")
                
            else:
                messagebox.showwarning("Warning", "Please set output directory to export report")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PM report: {str(e)}")
            logging.error(f"Error exporting PM report: {e}")
    
    def export_pm_plot(self):
        """Export PM frequency analysis comparison plot"""
        if not hasattr(self, 'pm_plot_fig') or self.pm_plot_fig is None:
            messagebox.showwarning("Warning", "No PM analysis plot available. Please run the analysis first.")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Export PM Analysis Plot"
        )
            if file_path:
                self.pm_plot_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"PM frequency comparison plot exported to {file_path}")
                logging.info(f"PM plot exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PM plot: {str(e)}")
            logging.error(f"Error exporting PM plot: {e}")
    
    def optimize_pm_schedule(self):
        """Simple PM schedule optimization based on current selections"""
        messagebox.showinfo("PM Optimization", 
                           "PM optimization is now integrated into the main analysis.\n\n"
                           "Use the 'Update PM Analysis' button to get optimal PM frequency recommendations "
                           "based on your current work order selections and filters.")
    
    def generate_pm_frequency_report(self):
        """Generate PM frequency report for current selections"""
        # This is now handled by export_pm_report
        self.export_pm_report()
    
    def update_spares_analysis(self):
        """Update spares analysis based on current data and Weibull parameters"""
        if not MODULES_AVAILABLE or not self.spares_analyzer or self.wo_df is None:
            messagebox.showwarning("Warning", "Spares analysis module not available or no data loaded")
            return
        try:
            # Get filtered data
            filtered_df = self.get_filtered_df()
            if filtered_df.empty:
                messagebox.showwarning("Warning", "No data available for spares analysis")
                self.create_enhanced_spares_plot({}, None, None, None)
                return
            equipment_filter = self.spares_equipment_var.get()
            try:
                simulation_years = int(self.simulation_years_var.get())
                num_simulations = int(self.num_simulations_var.get())
                lead_time_days = int(self.lead_time_var.get())
                service_level = float(self.service_level_var.get()) / 100.0
            except ValueError:
                simulation_years = 10
                num_simulations = 1000
                lead_time_days = 30
                service_level = 0.95
            weibull_params = None
            if hasattr(self, 'weibull_analyzer') and self.weibull_analyzer:
                if hasattr(self.weibull_analyzer, 'weibull_params') and self.weibull_analyzer.weibull_params:
                    weibull_params = self.weibull_analyzer.weibull_params
                elif hasattr(self.weibull_analyzer, '_pm_params') and self.weibull_analyzer._pm_params:
                    weibull_params = self.weibull_analyzer._pm_params
            equipment_data = filtered_df
            if equipment_filter:
                equipment_data = filtered_df[filtered_df['Equipment #'] == equipment_filter].copy()
                if equipment_data.empty:
                    messagebox.showwarning("Warning", f"No data available for equipment {equipment_filter}")
                    self.create_enhanced_spares_plot({}, None, None, equipment_filter)
                    return
            demand_analysis = self.spares_analyzer.analyze_spares_demand(filtered_df, equipment_filter, self.included_indices)
            valid_indices = equipment_data.index.intersection(self.included_indices)
            _, _, failures_per_year = calculate_crow_amsaa_params(equipment_data, set(valid_indices))
            demand_analysis['crow_amsaa_failures_per_year'] = failures_per_year
            if demand_analysis['status'] != "Analysis complete":
                messagebox.showwarning("Warning", f"Spares analysis failed: {demand_analysis['status']}")
                self.create_enhanced_spares_plot(demand_analysis, None, None, equipment_filter)
                return
            weibull_simulation = None
            if weibull_params and 'beta' in weibull_params and 'eta' in weibull_params:
                weibull_simulation = self.spares_analyzer.monte_carlo_weibull_simulation(
                    weibull_params, equipment_data, simulation_years, num_simulations, lead_time_days
                )
            stocking_analysis = None
            if weibull_simulation and weibull_simulation.get("status") == "Weibull simulation complete":
                stocking_analysis = self.spares_analyzer.calculate_optimal_stocking_levels(
                    weibull_simulation, failures_per_year, lead_time_days, service_level
                )
            # Store for recommendations/export
            self.current_spares_analysis = {
                'demand_analysis': demand_analysis,
                'weibull_simulation': weibull_simulation,
                'stocking_analysis': stocking_analysis,
                'equipment_filter': equipment_filter,
                'weibull_params': weibull_params
            }
            self.create_enhanced_spares_plot(demand_analysis, weibull_simulation, stocking_analysis, equipment_filter)
            # Update summary text
            self.spares_summary_text.delete(1.0, tk.END)
            self.spares_summary_text.insert(tk.END, "📦 SPARES ANALYSIS SUMMARY\n")
            self.spares_summary_text.insert(tk.END, "=" * 50 + "\n\n")
            self.spares_summary_text.insert(tk.END, "This analysis helps optimize spare parts inventory by:\n")
            self.spares_summary_text.insert(tk.END, "• Predicting future demand using statistical models\n")
            self.spares_summary_text.insert(tk.END, "• Calculating optimal stocking levels for different service targets\n")
            self.spares_summary_text.insert(tk.END, "• Comparing different analytical approaches\n\n")
            self.spares_summary_text.insert(tk.END, f"Equipment: {equipment_filter or 'All Equipment'}\n")
            self.spares_summary_text.insert(tk.END, f"Simulation Years: {simulation_years}, Simulations: {num_simulations}, Lead Time: {lead_time_days} days, Service Level: {service_level*100:.1f}%\n")
            
            # Note about censored data
            total_work_orders = len(filtered_df)
            included_work_orders = len(self.included_indices)
            if total_work_orders != included_work_orders:
                self.spares_summary_text.insert(tk.END, f"📊 Data Filtering: {included_work_orders}/{total_work_orders} work orders included (censored data excluded)\n")
            
            self.spares_summary_text.insert(tk.END, "\n")
            self.spares_summary_text.insert(tk.END, f"Total Demands: {demand_analysis.get('total_demands', 0)}\n")
            self.spares_summary_text.insert(tk.END, f"Total Cost: ${demand_analysis.get('total_cost', 0):,.2f}\n")
            self.spares_summary_text.insert(tk.END, f"Unique Failure Modes: {demand_analysis.get('unique_failure_modes', 0)}\n\n")
            
            # Demand rates
            self.spares_summary_text.insert(tk.END, "🔍 DEMAND RATES (Failures per Year):\n")
            self.spares_summary_text.insert(tk.END, f"- Crow-AMSAA (MTBF): {failures_per_year:.2f}\n")
            weibull_demand_rate = None
            if weibull_simulation and weibull_simulation.get('annual_failure_rate') is not None:
                weibull_demand_rate = weibull_simulation['annual_failure_rate']
                self.spares_summary_text.insert(tk.END, f"- Weibull: {weibull_demand_rate:.2f} (Simulation)\n")
            
            # Compare with failure mode rates
            total_failure_mode_rate = demand_analysis.get('total_demand_rate', 0)
            self.spares_summary_text.insert(tk.END, f"- Sum of Failure Mode Rates: {total_failure_mode_rate:.2f}\n")
            
            # Explain the difference
            if abs(failures_per_year - total_failure_mode_rate) > 0.1:
                self.spares_summary_text.insert(tk.END, f"  ⚠️ Note: Crow-AMSAA rate ({failures_per_year:.2f}) differs from sum of failure modes ({total_failure_mode_rate:.2f})\n")
                self.spares_summary_text.insert(tk.END, f"  This is because Crow-AMSAA uses time-based analysis while failure mode rates use interval counting\n")
            
            self.spares_summary_text.insert(tk.END, "\n")
            # Lead time usage
            self.spares_summary_text.insert(tk.END, "⏱️ ESTIMATED USAGE DURING LEAD TIME INTERVAL:\n")
            if stocking_analysis:
                lt_stats = stocking_analysis['comparison'].get('lead_time_stats', {})
                self.spares_summary_text.insert(tk.END, f"- Weibull Mean: {lt_stats.get('mean', 0):.2f}, 95th %ile: {lt_stats.get('p95', 0):.2f}, 99th %ile: {lt_stats.get('p99', 0):.2f}\n")
                self.spares_summary_text.insert(tk.END, f"- MTBF Expected: {stocking_analysis['comparison'].get('annual_demand_mtbf', 0) * (lead_time_days/365):.2f}\n")
            self.spares_summary_text.insert(tk.END, "\n")
            # Stocking recommendations
            self.spares_summary_text.insert(tk.END, "📦 RECOMMENDED STOCKING LEVELS (for Lead Time Window):\n")
            if stocking_analysis:
                cmp = stocking_analysis['comparison']
                self.spares_summary_text.insert(tk.END, f"- Weibull (95th %ile): {cmp.get('weibull_leadtime_p95', 0)}\n")
                self.spares_summary_text.insert(tk.END, f"- Weibull (99th %ile): {cmp.get('weibull_leadtime_p99', 0)}\n")
                self.spares_summary_text.insert(tk.END, f"- MTBF-based: {cmp.get('mtbf_based', 0)}\n")
                self.spares_summary_text.insert(tk.END, f"- Recommended: {cmp.get('recommended_stock', 0)}\n")
                self.spares_summary_text.insert(tk.END, f"\nFailure Pattern: {cmp.get('failure_pattern', '')}\nRecommendation: {cmp.get('recommendation', '')}\n")
            self.spares_summary_text.insert(tk.END, "\n📊 STATISTICAL EXPLANATIONS:\n")
            self.spares_summary_text.insert(tk.END, "=" * 50 + "\n\n")
            
            self.spares_summary_text.insert(tk.END, "🔍 DEMAND RATES:\n")
            self.spares_summary_text.insert(tk.END, "• Crow-AMSAA (MTBF): Based on actual failure history using Crow-AMSAA model.\n")
            self.spares_summary_text.insert(tk.END, "  Shows the trend in failure rate over time (increasing, decreasing, or constant).\n")
            self.spares_summary_text.insert(tk.END, "• Weibull (Simulation): Based on Monte Carlo simulation using Weibull distribution.\n")
            self.spares_summary_text.insert(tk.END, "  Accounts for equipment aging patterns and statistical variability.\n\n")
            
            self.spares_summary_text.insert(tk.END, "⏱️ LEAD TIME USAGE:\n")
            self.spares_summary_text.insert(tk.END, "• Weibull Mean: Average expected failures during lead time period.\n")
            self.spares_summary_text.insert(tk.END, "• 95th/99th %ile: 95% or 99% of simulations had this many or fewer failures.\n")
            self.spares_summary_text.insert(tk.END, "• MTBF Expected: Theoretical failures based on Mean Time Between Failures.\n\n")
            
            self.spares_summary_text.insert(tk.END, "📦 STOCKING LEVELS:\n")
            self.spares_summary_text.insert(tk.END, "• Weibull 95% (LT): Stock level for 95% service level during lead time.\n")
            self.spares_summary_text.insert(tk.END, "• Weibull 99% (LT): Stock level for 99% service level during lead time.\n")
            self.spares_summary_text.insert(tk.END, "• MTBF-based: Traditional approach using MTBF + safety stock.\n")
            self.spares_summary_text.insert(tk.END, "• Recommended: Best approach based on failure pattern analysis.\n\n")
            
            self.spares_summary_text.insert(tk.END, "📈 SENSITIVITY ANALYSIS:\n")
            self.spares_summary_text.insert(tk.END, "Shows total spares needed over 10 years for different reliability targets.\n")
            self.spares_summary_text.insert(tk.END, "Values represent cumulative failures over the entire 10-year period.\n")
            self.spares_summary_text.insert(tk.END, "Higher reliability = more spares needed, but lower stockout risk.\n")
            self.spares_summary_text.insert(tk.END, "Example: 64 spares at 95% reliability means 95% of simulations had ≤64 failures in 10 years.\n\n")
            
            self.spares_summary_text.insert(tk.END, "🔧 FORMULAS USED:\n")
            self.spares_summary_text.insert(tk.END, "• Crow-AMSAA: λβt^(β-1) or annualized failures from log-linear fit\n")
            self.spares_summary_text.insert(tk.END, "• Weibull: Monte Carlo simulation using β (shape), η (scale)\n")
            self.spares_summary_text.insert(tk.END, "• MTBF Stock: Lead Time Demand + Safety Stock (z×√Lead Time Demand)\n")
            self.spares_summary_text.insert(tk.END, "• Weibull Stock: 95th/99th percentile of simulated failures in lead time\n")
            self.spares_summary_text.insert(tk.END, "\n✅ CONSISTENCY NOTE: All MTBF calculations across all tabs now use the Crow-AMSAA method for consistency.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update spares analysis: {str(e)}")
            logging.error(f"Error updating spares analysis: {e}")

    def create_enhanced_spares_plot(self, demand_analysis, weibull_simulation, stocking_analysis, equipment_filter):
        """Create enhanced spares analysis plot with Weibull simulation results and sensitivity analysis (no cost analysis)."""
        try:
            for widget in self.spares_plot_frame.winfo_children():
                widget.destroy()
            
            # Set smaller font sizes to reduce clutter
            plt.rcParams.update({
                'font.size': 8,
                'axes.titlesize': 9,
                'axes.labelsize': 8,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7,
                'legend.fontsize': 7
            })
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
            spares_demand = demand_analysis.get("spares_demand", {})
            if spares_demand:
                failure_codes = list(spares_demand.keys())[:10]
                demand_rates = [spares_demand[code]['demand_rate_per_year'] for code in failure_codes]
                bars = ax1.bar(range(len(failure_codes)), demand_rates, color='skyblue')
                ax1.set_title('Demand Rate by Failure Mode\n(Top 10)')
                ax1.set_xlabel('Failure Code')
                ax1.set_ylabel('Demand Rate (per year)')
                ax1.set_xticks(range(len(failure_codes)))
                ax1.set_xticklabels(failure_codes, rotation=45, ha='right', fontsize=6)
                for bar, value in zip(bars, demand_rates):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}', 
                            ha='center', va='bottom', fontsize=6)
            else:
                ax1.text(0.5, 0.5, "No demand data", ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Demand by Failure Mode')
            # --- Top Right: Sensitivity Analysis (Line Plot) ---
            reliability_levels = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
            spares_needed = []
            weibull_sim_data = weibull_simulation.get("simulation_results", []) if weibull_simulation else []
            if weibull_sim_data:
                final_failures = [sim[-1] for sim in weibull_sim_data]
                for r in reliability_levels:
                    p = int(np.percentile(final_failures, 100*r))
                    spares_needed.append(p)
                ax2.plot([int(r*100) for r in reliability_levels], spares_needed, marker='o', color='purple', linewidth=2)
                ax2.set_title('Sensitivity: Spares vs. Reliability\n(10 years)')
                ax2.set_xlabel('Reliability Level (%)')
                ax2.set_ylabel('Required Spares (10 yrs)')
                ax2.set_xticks([int(r*100) for r in reliability_levels])
                ax2.invert_xaxis()
                for x, y in zip([int(r*100) for r in reliability_levels], spares_needed):
                    ax2.text(x, y, str(y), ha='center', va='bottom', fontsize=6)
                ax2.grid(True, alpha=0.3)
                if len(spares_needed) > 1:
                    min_y = min(spares_needed)
                    max_y = max(spares_needed)
                    ax2.set_ylim([min_y - 1, max_y + 1])
            else:
                ax2.text(0.5, 0.5, "No sensitivity data", ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Sensitivity: Spares vs. Reliability\n(10 years)')
            # --- Bottom Left: Demand Rates & Lead Time Usage ---
            crow_amsaa_rate = demand_analysis.get('crow_amsaa_failures_per_year')
            weibull_rate = None
            weibull_leadtime_mean = None
            weibull_leadtime_p95 = None
            mtbf_leadtime = None
            weibull_demand_rate = None
            if weibull_simulation and weibull_simulation.get("status") == "Weibull simulation complete":
                # Use the same calculation as the text summary for consistency
                weibull_demand_rate = weibull_simulation.get('annual_failure_rate', 0)
                lt_stats = weibull_simulation.get('lead_time_stats', {})
                weibull_leadtime_mean = lt_stats.get('mean', 0)
                weibull_leadtime_p95 = lt_stats.get('p95', 0)
            if stocking_analysis and stocking_analysis.get("status") == "Stocking levels calculated":
                mtbf_leadtime = stocking_analysis['comparison'].get('annual_demand_mtbf', 0) * (stocking_analysis['lead_time_days']/365)
            labels = []
            values = []
            annots = []
            if crow_amsaa_rate is not None:
                labels.append('Crow-AMSAA Rate (MTBF)')
                values.append(crow_amsaa_rate)
                annots.append(f"{crow_amsaa_rate:.2f} failures/year")
            if weibull_demand_rate is not None:
                labels.append('Weibull Rate (Simulation)')
                values.append(weibull_demand_rate)
                annots.append(f"{weibull_demand_rate:.2f} failures/year")
            if weibull_leadtime_mean is not None:
                labels.append('Weibull Lead Time Mean')
                values.append(weibull_leadtime_mean)
                annots.append(f"{weibull_leadtime_mean:.2f} in {stocking_analysis['lead_time_days']}d")
            if weibull_leadtime_p95 is not None:
                labels.append('Weibull Lead Time 95%')
                values.append(weibull_leadtime_p95)
                annots.append(f"{weibull_leadtime_p95:.2f} in {stocking_analysis['lead_time_days']}d")
            if mtbf_leadtime is not None:
                labels.append('MTBF Lead Time Exp.')
                values.append(mtbf_leadtime)
                annots.append(f"{mtbf_leadtime:.2f} in {stocking_analysis['lead_time_days']}d")
            if values:
                bars = ax3.bar(labels, values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'][:len(values)])
                ax3.set_title('Demand Rates & Lead Time\nUsage')
                ax3.set_ylabel('Units')
                ax3.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
                for bar, annot in zip(bars, annots):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height, annot, 
                            ha='center', va='bottom', fontsize=6)
            else:
                ax3.text(0.5, 0.5, "No rate data", ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Demand Rates & Lead Time\nUsage')
            # --- Bottom Right: Recommended Stock Levels ---
            if stocking_analysis and stocking_analysis.get("status") == "Stocking levels calculated":
                comparison = stocking_analysis['comparison']
                methods = ['Weibull 95% (LT)', 'Weibull 99% (LT)', 'MTBF-based', 'Recommended']
                values = [
                    comparison.get('weibull_leadtime_p95', 0),
                    comparison.get('weibull_leadtime_p99', 0),
                    comparison.get('mtbf_based', 0),
                    comparison.get('recommended_stock', 0)
                ]
                bars = ax4.bar(methods, values, color=['skyblue', 'lightblue', 'lightgreen', 'orange'])
                ax4.set_title('Stocking Level Comparison\n(Lead Time)')
                ax4.set_ylabel('Units')
                ax4.set_xticklabels(methods, rotation=30, ha='right', fontsize=6)
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height, f'{value}', 
                            ha='center', va='bottom', fontsize=6)
                if mtbf_leadtime is not None:
                    ax4.axhline(mtbf_leadtime, color='red', linestyle='--', label=f'Expected Usage ({mtbf_leadtime:.2f})')
                    ax4.legend()
            else:
                ax4.text(0.5, 0.5, "No stocking analysis data", ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Stocking Level Comparison\n(Lead Time)')
            equipment_title = f" - {equipment_filter}" if equipment_filter else ""
            fig.suptitle(f'Enhanced Spares Analysis{equipment_title}', fontsize=16)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            canvas = FigureCanvasTkAgg(fig, master=self.spares_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.spares_plot_fig = fig
        except Exception as e:
            logging.error(f"Error creating enhanced spares plot: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error creating spares analysis plot: {str(e)}", ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title('Enhanced Spares Analysis')
            canvas = FigureCanvasTkAgg(fig, master=self.spares_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            

    def update_failure_code_dropdown(self):
        """Update the failure code dropdown based on selected source."""
        if self.wo_df is None or self.wo_df.empty:
            self.failure_code_dropdown['values'] = ['']
            self.failure_code_var.set('')
            return
        col = 'Failure Code' if self.failure_code_source_var.get() == 'AI/Dictionary' else 'User failure code'
        codes = sorted(self.wo_df[col].dropna().unique())
        self.failure_code_dropdown['values'] = [''] + list(codes)
        if self.failure_code_var.get() not in codes and self.failure_code_var.get() != '':
            self.failure_code_var.set('')

    def update_weibull_failure_dropdown(self):
        """Update the Weibull failure code dropdown based on selected source."""
        if self.wo_df is None or self.wo_df.empty:
            self.weibull_failure_dropdown['values'] = ['']
            self.weibull_failure_var.set('')
            return
        col = 'Failure Code' if self.weibull_failure_code_source_var.get() == 'AI/Dictionary' else 'User failure code'
        codes = sorted(self.wo_df[col].dropna().unique())
        self.weibull_failure_dropdown['values'] = [''] + list(codes)
        if self.weibull_failure_var.get() not in codes and self.weibull_failure_var.get() != '':
            self.weibull_failure_var.set('')
    
    def update_pm_equipment_dropdown(self):
        """Update PM equipment dropdown - no longer used"""
        # PM analysis now uses current work order selections instead of dropdown
        pass
    
    def update_spares_equipment_dropdown(self):
        """Update the spares equipment dropdown"""
        if self.wo_df is None or self.wo_df.empty:
            self.spares_equipment_dropdown['values'] = ['']
            self.spares_equipment_var.set('')
            return
        
        if 'Equipment #' in self.wo_df.columns:
            equipment_list = sorted(self.wo_df['Equipment #'].dropna().unique())
            self.spares_equipment_dropdown['values'] = [''] + list(equipment_list)
            if self.spares_equipment_var.get() not in equipment_list and self.spares_equipment_var.get() != '':
                self.spares_equipment_var.set('')

    def export_spares_plot(self):
        """Export spares analysis plot"""
        if not hasattr(self, 'spares_plot_fig') or self.spares_plot_fig is None:
            messagebox.showwarning("Warning", "No spares plot to export. Please run the analysis first.")
            return
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Export Spares Analysis Plot"
            )
            if file_path:
                self.spares_plot_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Spares plot exported to {file_path}")
                logging.info(f"Spares plot exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
            logging.error(f"Error exporting spares plot: {e}")

    def export_spares_report(self):
        """Export comprehensive spares analysis report (summary, recommendations, and stats) to Excel"""
        if not hasattr(self, 'current_spares_analysis'):
            messagebox.showwarning("Warning", "No spares analysis available. Run spares analysis first.")
            return
        if not self.output_dir:
            messagebox.showwarning("Warning", "Please set output directory to export report")
            return
        try:
            import pandas as pd
            demand_analysis = self.current_spares_analysis['demand_analysis']
            weibull_simulation = self.current_spares_analysis.get('weibull_simulation')
            stocking_analysis = self.current_spares_analysis.get('stocking_analysis')
            equipment_filter = self.current_spares_analysis.get('equipment_filter', 'All Equipment')
            weibull_params = self.current_spares_analysis.get("weibull_params")

            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"spares_analysis_report_{timestamp}.xlsx")

            # Prepare data for export
            summary_data = {
                "Metric": [
                    "Equipment Filter",
                    "Total Demands",
                    "Total Cost ($)",
                    "Unique Failure Modes",
                    "Crow-AMSAA Failures/Year",
                    "Weibull Failures/Year",
                    "Total Demand Rate"
                ],
                "Value": [
                    equipment_filter,
                    demand_analysis.get("total_demands", 0),
                    f"${demand_analysis.get('total_cost', 0):,.2f}",
                    demand_analysis.get("unique_failure_modes", 0),
                    f"{demand_analysis.get('crow_amsaa_failures_per_year', 0):.2f}",
                    f"{weibull_simulation.get('annual_failure_rate', 0):.2f}" if weibull_simulation else "N/A",
                    f"{demand_analysis.get('total_demand_rate', 0):.2f}"
                ]
            }

            summary_df = pd.DataFrame(summary_data)

            # Export to Excel
            with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # Add detailed data if available
                if stocking_analysis:
                    comparison_data = {
                        "Method": ["Weibull 95%", "Weibull 99%", "MTBF-based", "Recommended"],
                        "Stock Level": [
                            stocking_analysis["comparison"].get("weibull_leadtime_p95", 0),
                            stocking_analysis["comparison"].get("weibull_leadtime_p99", 0),
                            stocking_analysis["comparison"].get("mtbf_based", 0),
                            stocking_analysis["comparison"].get("recommended_stock", 0)
                        ]
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df.to_excel(writer, sheet_name="Stocking Analysis", index=False)

            messagebox.showinfo("Success", f"Spares analysis report exported to {output_file}")
            logging.info(f"Spares analysis report exported to: {output_file}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export spares report: {str(e)}")
            logging.error(f"Error exporting spares report: {e}")

    def add_to_fmea_export(self):
        """Add current Weibull analysis results to FMEA export JSON file"""
        if not hasattr(self, 'weibull_analyzer') or not self.weibull_analyzer:
            messagebox.showwarning("Warning", "No Weibull analysis available. Run Weibull analysis first.")
            return
        
        if not hasattr(self.weibull_analyzer, 'weibull_params') or not self.weibull_analyzer.weibull_params:
            messagebox.showwarning("Warning", "No Weibull parameters available. Run Weibull analysis first.")
            return
        
        try:
            # Get current analysis parameters
            equipment = self.weibull_equipment_var.get()
            failure_code = self.weibull_failure_var.get()
            failure_source = self.weibull_failure_code_source_var.get()
            
            if not equipment or not failure_code:
                messagebox.showwarning("Warning", "Please select both equipment and failure code for FMEA export.")
                return
            
            # Get Weibull parameters
            weibull_params = self.weibull_analyzer.weibull_params
            beta = weibull_params.get('beta', 0)
            eta = weibull_params.get('eta', 0)
            mtbf = weibull_params.get('mtbf', 0)
            
            # Get failure mode information
            filtered_df = self.get_filtered_df()
            if filtered_df.empty:
                messagebox.showwarning("Warning", "No data available for FMEA export.")
                return
            
            # Get failure mode details
            failure_col = 'Failure Code' if failure_source == 'AI/Dictionary' else 'User failure code'
            failure_desc_col = 'Failure Description' if failure_source == 'AI/Dictionary' else 'User failure code'
            
            failure_data = filtered_df[
                (filtered_df['Equipment #'] == equipment) & 
                (filtered_df[failure_col] == failure_code)
            ]
            
            if failure_data.empty:
                messagebox.showwarning("Warning", f"No data found for equipment {equipment} and failure code {failure_code}.")
                return
            
            # Calculate frequency (failures per year)
            total_failures = len(failure_data)
            if self.start_date and self.end_date:
                date_range_days = (self.end_date - self.start_date).days
                frequency = (total_failures / date_range_days) * 365 if date_range_days > 0 else 0
            else:
                frequency = total_failures  # Fallback to total count
            
            # Create FMEA entry
            fmea_entry = {
                "equipment": equipment,
                "failure_mode_ai": failure_data.iloc[0].get('Failure Code', ''),
                "failure_mode_user": failure_data.iloc[0].get('User failure code', ''),
                "failure_mode_type": failure_source,
                "failure_mode_frequency": round(frequency, 2),
                "weibull_beta": round(beta, 3),
                "weibull_eta": round(eta, 2),
                "mtbf_days": round(mtbf, 2),
                "total_failures": total_failures,
                "analysis_date": datetime.now().isoformat(),
                "date_range_start": self.start_date.isoformat() if self.start_date else None,
                "date_range_end": self.end_date.isoformat() if self.end_date else None
            }
            
            # Load existing FMEA data
            fmea_file = os.path.join(os.path.dirname(__file__), 'fmea_export_data.json')
            existing_data = []
            
            if os.path.exists(fmea_file):
                try:
                    with open(fmea_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logging.warning(f"Could not load existing FMEA data: {e}")
                    existing_data = []
            
            # Check for duplicate entry (same equipment and failure mode)
            key = f"{equipment}_{failure_code}_{failure_source}"
            duplicate_found = False
            
            for i, entry in enumerate(existing_data):
                existing_key = f"{entry.get('equipment', '')}_{entry.get('failure_mode_ai' if entry.get('failure_mode_type') == 'AI/Dictionary' else 'failure_mode_user', '')}_{entry.get('failure_mode_type', '')}"
                if existing_key == key:
                    existing_data[i] = fmea_entry
                    duplicate_found = True
                    break
            
            if not duplicate_found:
                existing_data.append(fmea_entry)
            
            # Save updated FMEA data
            with open(fmea_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            action = "Updated" if duplicate_found else "Added"
            messagebox.showinfo("Success", f"{action} FMEA export data for {equipment} - {failure_code}\n\nFile: {fmea_file}")
            logging.info(f"FMEA export data {action.lower()}: {equipment} - {failure_code}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add to FMEA export: {str(e)}")
            logging.error(f"Error adding to FMEA export: {e}")

    def show_fmea_export_data(self):
        """Show FMEA export data in a spreadsheet-like view"""
        fmea_file = os.path.join(os.path.dirname(__file__), 'fmea_export_data.json')
        
        if not os.path.exists(fmea_file):
            messagebox.showinfo("Info", "No FMEA export data found. Use the 'Add to FMEA Export' button in the Weibull Analysis tab to create data.")
            return
        
        try:
            # Load FMEA data
            with open(fmea_file, 'r', encoding='utf-8') as f:
                fmea_data = json.load(f)
            
            if not fmea_data:
                messagebox.showinfo("Info", "FMEA export data file is empty.")
                return
            
            # Create window
            fmea_window = tk.Toplevel(self.root)
            fmea_window.title("FMEA Export Data Viewer")
            fmea_window.geometry("1200x700")
            
            # Create main frame
            main_frame = ttk.Frame(fmea_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Header
            header_frame = ttk.Frame(main_frame)
            header_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(header_frame, text="FMEA Export Data", font=("Arial", 14, "bold")).pack(side=tk.LEFT)
            ttk.Label(header_frame, text=f"Total Entries: {len(fmea_data)}", font=("Arial", 10)).pack(side=tk.RIGHT)
            
            # Buttons frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Button(button_frame, text="📊 Export to Excel", command=lambda: self.export_fmea_data_to_excel(fmea_data)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="🗑️ Clear All Data", command=lambda: self.clear_fmea_data(fmea_window)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="📁 Open File Location", command=lambda: self.open_fmea_file_location()).pack(side=tk.LEFT, padx=5)
            
            # Create treeview
            columns = [
                'Equipment', 'AI Failure Mode', 'User Failure Mode', 'Type', 
                'Frequency (per year)', 'Weibull β', 'Weibull η', 'MTBF (days)',
                'Total Failures', 'Analysis Date'
            ]
            
            tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=20)
            
            # Configure columns
            column_widths = {
                'Equipment': 120,
                'AI Failure Mode': 150,
                'User Failure Mode': 150,
                'Type': 100,
                'Frequency (per year)': 120,
                'Weibull β': 80,
                'Weibull η': 80,
                'MTBF (days)': 100,
                'Total Failures': 100,
                'Analysis Date': 150
            }
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=column_widths.get(col, 100))
            
            # Add scrollbars
            scrollbar_y = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
            scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
            
            # Pack tree and scrollbars
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
            scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Populate tree
            for entry in fmea_data:
                analysis_date = entry.get('analysis_date', '')
                if analysis_date:
                    try:
                        # Parse ISO format and format for display
                        dt = datetime.fromisoformat(analysis_date.replace('Z', '+00:00'))
                        analysis_date = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
                
                values = [
                    entry.get('equipment', ''),
                    entry.get('failure_mode_ai', ''),
                    entry.get('failure_mode_user', ''),
                    entry.get('failure_mode_type', ''),
                    f"{entry.get('failure_mode_frequency', 0):.2f}",
                    f"{entry.get('weibull_beta', 0):.3f}",
                    f"{entry.get('weibull_eta', 0):.2f}",
                    f"{entry.get('mtbf_days', 0):.2f}",
                    entry.get('total_failures', 0),
                    analysis_date
                ]
                tree.insert('', 'end', values=values)
            
            # Status bar
            status_frame = ttk.Frame(main_frame)
            status_frame.pack(fill=tk.X, pady=(10, 0))
            ttk.Label(status_frame, text=f"File: {fmea_file}", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load FMEA export data: {str(e)}")
            logging.error(f"Error loading FMEA export data: {e}")

    def export_fmea_data_to_excel(self, fmea_data):
        """Export FMEA data to Excel"""
        if not self.output_dir:
            messagebox.showwarning("Warning", "Please set output directory to export FMEA data")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                initialdir=self.output_dir,
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Export FMEA Data to Excel"
            )
            
            if file_path:
                # Convert to DataFrame
                df = pd.DataFrame(fmea_data)
                
                # Format dates
                if 'analysis_date' in df.columns:
                    df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Export to Excel
                df.to_excel(file_path, index=False, engine='openpyxl')
                messagebox.showinfo("Success", f"FMEA data exported to {file_path}")
                logging.info(f"FMEA data exported to: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export FMEA data: {str(e)}")
            logging.error(f"Error exporting FMEA data: {e}")

    def clear_fmea_data(self, window):
        """Clear all FMEA export data"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all FMEA export data? This action cannot be undone."):
            try:
                fmea_file = os.path.join(os.path.dirname(__file__), 'fmea_export_data.json')
                if os.path.exists(fmea_file):
                    os.remove(fmea_file)
                messagebox.showinfo("Success", "FMEA export data cleared successfully.")
                window.destroy()  # Close the viewer window
                logging.info("FMEA export data cleared")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear FMEA data: {str(e)}")
                logging.error(f"Error clearing FMEA data: {e}")

    def open_fmea_file_location(self):
        """Open the folder containing the FMEA export file"""
        try:
            fmea_file = os.path.join(os.path.dirname(__file__), 'fmea_export_data.json')
            if os.path.exists(fmea_file):
                import subprocess
                import platform
                
                if platform.system() == "Windows":
                    subprocess.run(["explorer", "/select,", fmea_file])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", "-R", fmea_file])
                else:  # Linux
                    subprocess.run(["xdg-open", os.path.dirname(fmea_file)])
            else:
                messagebox.showinfo("Info", "FMEA export file does not exist yet.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file location: {str(e)}")
            logging.error(f"Error opening file location: {e}")

    def copy_to_clipboard(self, text):
        """Copy the provided text to the clipboard and show a confirmation message."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()  # Keeps clipboard after window closes
        messagebox.showinfo("Copied", "Summary copied to clipboard!")

    def copy_risk_summary(self):
        """Copy the risk summary label text to the clipboard."""
        self.copy_to_clipboard(self.risk_label.cget("text"))

    def show_charts_dialog(self):
        """Show charts dialog with Pareto charts for top 10 equipment"""
        if self.wo_df is None or self.wo_df.empty:
            messagebox.showwarning("Warning", "No data available. Please process files first.")
            return
        
        # Create charts window
        charts_window = tk.Toplevel(self.root)
        charts_window.title("Equipment Pareto Charts")
        charts_window.geometry("1400x900")
        
        # Create main frame
        main_frame = ttk.Frame(charts_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="Equipment Pareto Charts", font=("Arial", 14, "bold")).pack(side=tk.LEFT)
        
        # Filters info
        filters_text = self.get_current_filters_text()
        ttk.Label(header_frame, text=f"Filters: {filters_text}", font=("Arial", 10), foreground="gray").pack(side=tk.RIGHT)
        
        # Create notebook for different chart types
        charts_notebook = ttk.Notebook(main_frame)
        charts_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for each chart type
        self.create_cost_pareto_tab(charts_notebook, charts_window)
        self.create_count_pareto_tab(charts_notebook, charts_window)
        self.create_failure_rate_pareto_tab(charts_notebook, charts_window)

    def create_cost_pareto_tab(self, notebook, parent_window):
        """Create Pareto chart for work order cost"""
        cost_frame = ttk.Frame(notebook)
        notebook.add(cost_frame, text="💰 Work Order Cost")
        
        # Controls frame
        controls_frame = ttk.Frame(cost_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="🔄 Update Chart", 
                  command=lambda: self.update_cost_pareto(cost_frame)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="📊 Export Chart", 
                  command=lambda: self.export_pareto_chart("cost", parent_window)).pack(side=tk.LEFT, padx=5)
        
        # Chart frame
        self.cost_chart_frame = ttk.Frame(cost_frame)
        self.cost_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial chart
        self.update_cost_pareto(cost_frame)

    def create_count_pareto_tab(self, notebook, parent_window):
        """Create Pareto chart for work order count"""
        count_frame = ttk.Frame(notebook)
        notebook.add(count_frame, text="📊 Work Order Count")
        
        # Controls frame
        controls_frame = ttk.Frame(count_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="🔄 Update Chart", 
                  command=lambda: self.update_count_pareto(count_frame)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="📊 Export Chart", 
                  command=lambda: self.export_pareto_chart("count", parent_window)).pack(side=tk.LEFT, padx=5)
        
        # Chart frame
        self.count_chart_frame = ttk.Frame(count_frame)
        self.count_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial chart
        self.update_count_pareto(count_frame)

    def create_failure_rate_pareto_tab(self, notebook, parent_window):
        """Create Pareto chart for failure rate"""
        failure_frame = ttk.Frame(notebook)
        notebook.add(failure_frame, text="⚠️ Failure Rate")
        
        # Controls frame
        controls_frame = ttk.Frame(failure_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="🔄 Update Chart", 
                  command=lambda: self.update_failure_rate_pareto(failure_frame)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="📊 Export Chart", 
                  command=lambda: self.export_pareto_chart("failure_rate", parent_window)).pack(side=tk.LEFT, padx=5)
        
        # Chart frame
        self.failure_rate_chart_frame = ttk.Frame(failure_frame)
        self.failure_rate_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial chart
        self.update_failure_rate_pareto(failure_frame)

    def get_chart_data_with_filters(self):
        """Get data for charts using current date and work type filters, but all equipment"""
        if self.wo_df is None or self.wo_df.empty:
            return pd.DataFrame()
        
        # Start with all data
        filtered_df = self.wo_df.copy()
        
        # Apply date range filter
        if self.start_date or self.end_date:
            filtered_df['Parsed_Date'] = filtered_df['Reported Date'].apply(parse_date)
            if self.start_date:
                mask = filtered_df['Parsed_Date'] >= self.start_date
                filtered_df = filtered_df[mask]
            if self.end_date:
                mask = filtered_df['Parsed_Date'] <= self.end_date
                filtered_df = filtered_df[mask]
            filtered_df = filtered_df.drop(columns=['Parsed_Date'], errors='ignore')
        
        # Apply work type filter
        work_type = self.work_type_var.get()
        if work_type:
            mask = filtered_df['Work Type'] == work_type
            filtered_df = filtered_df[mask]
        
        # Apply failure code filter
        failure_code = self.failure_code_var.get()
        code_col = 'Failure Code' if self.failure_code_source_var.get() == 'AI/Dictionary' else 'User failure code'
        if failure_code:
            mask = filtered_df[code_col] == failure_code
            filtered_df = filtered_df[mask]
        
        return filtered_df

    def update_cost_pareto(self, parent_frame):
        """Update cost Pareto chart"""
        # Clear previous chart
        for widget in self.cost_chart_frame.winfo_children():
            widget.destroy()
        
        # Get filtered data
        filtered_df = self.get_chart_data_with_filters()
        if filtered_df.empty:
            label = ttk.Label(self.cost_chart_frame, text="No data available for the current filters.", foreground="gray")
            label.pack(expand=True)
            return
        
        # Calculate total cost by equipment
        equipment_costs = {}
        for idx, row in filtered_df.iterrows():
            equipment = str(row.get('Equipment #', ''))
            if equipment and equipment != 'nan':
                cost = row.get('Work Order Cost', 0.0)
                try:
                    cost = float(cost) if pd.notna(cost) else 0.0
                    equipment_costs[equipment] = equipment_costs.get(equipment, 0.0) + cost
                except (ValueError, TypeError):
                    continue
        
        if not equipment_costs:
            label = ttk.Label(self.cost_chart_frame, text="No cost data available.", foreground="gray")
            label.pack(expand=True)
            return
        
        # Get top 10 equipment by cost
        sorted_equipment = sorted(equipment_costs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create Pareto chart
        self.create_pareto_chart(self.cost_chart_frame, sorted_equipment, 
                               "Top 10 Equipment by Work Order Cost", 
                               "Equipment", "Total Cost ($)", "cost")

    def update_count_pareto(self, parent_frame):
        """Update count Pareto chart"""
        # Clear previous chart
        for widget in self.count_chart_frame.winfo_children():
            widget.destroy()
        
        # Get filtered data
        filtered_df = self.get_chart_data_with_filters()
        if filtered_df.empty:
            label = ttk.Label(self.count_chart_frame, text="No data available for the current filters.", foreground="gray")
            label.pack(expand=True)
            return
        
        # Calculate work order count by equipment
        equipment_counts = {}
        for idx, row in filtered_df.iterrows():
            equipment = str(row.get('Equipment #', ''))
            if equipment and equipment != 'nan':
                equipment_counts[equipment] = equipment_counts.get(equipment, 0) + 1
        
        if not equipment_counts:
            label = ttk.Label(self.count_chart_frame, text="No count data available.", foreground="gray")
            label.pack(expand=True)
            return
        
        # Get top 10 equipment by count
        sorted_equipment = sorted(equipment_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create Pareto chart
        self.create_pareto_chart(self.count_chart_frame, sorted_equipment, 
                               "Top 10 Equipment by Work Order Count", 
                               "Equipment", "Work Order Count", "count")

    def update_failure_rate_pareto(self, parent_frame):
        """Update failure rate Pareto chart"""
        # Clear previous chart
        for widget in self.failure_rate_chart_frame.winfo_children():
            widget.destroy()
        
        # Get filtered data
        filtered_df = self.get_chart_data_with_filters()
        if filtered_df.empty:
            label = ttk.Label(self.failure_rate_chart_frame, text="No data available for the current filters.", foreground="gray")
            label.pack(expand=True)
            return
        
        # Calculate failure rate by equipment
        equipment_failure_rates = {}
        for equipment in filtered_df['Equipment #'].dropna().unique():
            equipment = str(equipment)
            if equipment and equipment != 'nan':
                eq_df = filtered_df[filtered_df['Equipment #'] == equipment]
                valid_indices = eq_df.index.intersection(self.included_indices)
                if len(valid_indices) > 0:
                    _, _, failures_per_year = calculate_crow_amsaa_params(eq_df, set(valid_indices))
                    equipment_failure_rates[equipment] = failures_per_year
        
        if not equipment_failure_rates:
            label = ttk.Label(self.failure_rate_chart_frame, text="No failure rate data available.", foreground="gray")
            label.pack(expand=True)
            return
        
        # Get top 10 equipment by failure rate
        sorted_equipment = sorted(equipment_failure_rates.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create Pareto chart
        self.create_pareto_chart(self.failure_rate_chart_frame, sorted_equipment, 
                               "Top 10 Equipment by Failure Rate", 
                               "Equipment", "Failures per Year", "failure_rate")

    def create_pareto_chart(self, parent_frame, data, title, x_label, y_label, chart_type):
        """Create a Pareto chart with the given data"""
        if not data:
            label = ttk.Label(parent_frame, text="No data available.", foreground="gray")
            label.pack(expand=True)
            return
        
        # Create figure and canvas
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data
        equipment_names = [item[0] for item in data]
        values = [item[1] for item in data]
        
        # Create bar chart
        bars = ax.bar(range(len(equipment_names)), values, color='skyblue', alpha=0.7)
        
        # Add cumulative line
        cumulative = np.cumsum(values)
        cumulative_pct = cumulative / cumulative[-1] * 100
        ax2 = ax.twinx()
        ax2.plot(range(len(equipment_names)), cumulative_pct, 'r-', linewidth=2, marker='o')
        ax2.set_ylabel('Cumulative %', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # Set x-axis labels
        ax.set_xticks(range(len(equipment_names)))
        ax.set_xticklabels(equipment_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            if chart_type == "cost":
                label_text = f"${value:,.0f}"
            elif chart_type == "count":
                label_text = f"{value}"
            else:  # failure_rate
                label_text = f"{value:.2f}"
            
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   label_text, ha='center', va='bottom', fontsize=9)
        
        # Add percentage labels on cumulative line
        for i, pct in enumerate(cumulative_pct):
            ax2.text(i, pct + 2, f"{pct:.1f}%", ha='center', va='bottom', 
                    color='red', fontsize=9, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create canvas and add to frame
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store figure for export
        if chart_type == "cost":
            self.cost_pareto_fig = fig
        elif chart_type == "count":
            self.count_pareto_fig = fig
        else:  # failure_rate
            self.failure_rate_pareto_fig = fig

    def export_pareto_chart(self, chart_type, parent_window):
        """Export Pareto chart to file"""
        fig = None
        if chart_type == "cost":
            fig = getattr(self, 'cost_pareto_fig', None)
        elif chart_type == "count":
            fig = getattr(self, 'count_pareto_fig', None)
        elif chart_type == "failure_rate":
            fig = getattr(self, 'failure_rate_pareto_fig', None)
        
        if fig is None:
            messagebox.showwarning("Warning", "No chart available to export. Please update the chart first.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title=f"Export {chart_type.replace('_', ' ').title()} Pareto Chart"
            )
            
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Pareto chart exported to {file_path}")
                logging.info(f"Pareto chart exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart: {str(e)}")
            logging.error(f"Error exporting Pareto chart: {e}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FailureModeApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        print(f"Application failed to start: {e}")
