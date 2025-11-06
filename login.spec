# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import os

# List of directories to exclude
exclude_dirs = ['train', 'ISO-Evaluation', 'dist', 'build', 'forecase_maps', 'forecast_plots']

# Get all Python files in the main directory
main_py_files = [(f, '.') for f in os.listdir('.') if f.endswith('.py')]

# Get all directories in the current path
all_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d not in exclude_dirs]

# Initialize data files list
datas = main_py_files  # Add all Python files from main directory

# Add all directories to datas
for dir_name in all_dirs:
    datas.append((dir_name, dir_name))

# Add specific files and directories that need to be included
datas.extend([
    ('CSV/merged_stations.xlsx', 'CSV'),
    ('CSV/merged_stations_mag.xlsx', 'CSV'),
    ('CSV/chlorophyll_predictions_by_station.csv', 'CSV'),
    ('heatmapper/*.geojson', 'heatmapper'),
    ('heatmapper/heatmapper.py', 'heatmapper'),
    ('heatmapper/*', 'heatmapper'),
    ('pkl/*.pkl', 'pkl'),
    ('Icons', 'Icons'),
    ('user_data_directory', 'user_data_directory'),
    ('DevPics', 'DevPics'),
])

binaries = []
hiddenimports = [
    # Existing imports...
    'main',
    'sidebar',
    'dashboard',
    'about',
    'inputData',
    'waterQualityReport',
    'prediction',
    'settings',
    'icons',
    'globals',
    'utils',
    'audit',
    'chlorophyll_forecaster',
    'model_trainer',
    'heatmapper',
    'heatmapper.Heatmapper',
    'branca',
    'branca.colormap',

    # Additional imports from model_trainer.py
    'scipy.stats.skew',
    'numpy.log1p',
    'sklearn.model_selection.train_test_split',

    # Visualization related
    'matplotlib.figure',
    'matplotlib.axes',
    'seaborn.histplot',
    'seaborn.heatmap',
    'seaborn.regplot',

    # Data handling
    'pandas.to_datetime',
    'pandas.DataFrame',
    'pandas.Series',
    'numpy.issubdtype',
    'numpy.number',

    # File operations
    'pickle.dump',
    'pickle.load',
    'os.path',

    # Statistical and ML
    'sklearn.compose',
    'sklearn.compose.ColumnTransformer',
    'sklearn.feature_selection',
    'sklearn.feature_selection.SelectFromModel',
    'sklearn.preprocessing.StandardScaler',

    # Existing imports continue...
    'PIL._tkinter_finder',
    'webview',
    'pandas',
    'numpy',
    'matplotlib',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.pyplot',
    'seaborn',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'cryptography',
    'cryptography.fernet',
    'pickle',
    'datetime',
    'dateutil',
    'dateutil.relativedelta',
    'warnings',
    'os',
    'json',
    're',
    'base64',
    'ctypes',
    'csv',
    'pathlib',
    'geopandas',
    'folium',
    'folium.plugins',
    'folium.plugins.HeatMap',
    'folium.plugins.HeatMapWithTime',
    'sklearn',
    'sklearn.preprocessing',
    'sklearn.preprocessing.RobustScaler',
    'sklearn.ensemble',
    'sklearn.ensemble.RandomForestRegressor',
    'sklearn.ensemble.GradientBoostingRegressor',
    'sklearn.model_selection',
    'sklearn.model_selection.TimeSeriesSplit',
    'sklearn.model_selection.learning_curve',
    'sklearn.metrics',
    'sklearn.metrics.mean_squared_error',
    'sklearn.metrics.mean_absolute_error',
    'sklearn.metrics.r2_score',
    'sklearn.pipeline',
    'sklearn.pipeline.Pipeline',
    'scipy',
    'scipy.stats',
    'scipy.stats.pearsonr',
    'skopt',
    'skopt.BayesSearchCV',
    'skopt.space',
    'skopt.space.Real',
    'skopt.space.Integer'

    # Model training specific imports
    'sklearn.base',
    'sklearn.utils',
    'sklearn.utils.validation',
    'sklearn.exceptions',
    'sklearn.utils.multiclass',
    'sklearn.preprocessing._data',
    'sklearn.ensemble._forest',
    'sklearn.tree',
    'sklearn.tree._tree',
    'sklearn.utils._param_validation',
    'sklearn.model_selection._split',
    'sklearn.utils.fixes',
    'sklearn.metrics._regression',
    'sklearn.metrics._scorer',

    # File handling for model
    'joblib',
    'tempfile',
    'shutil',

    # Threading and process handling
    'threading',
    'queue',
    'multiprocessing',
    'concurrent.futures',

    # Progress tracking
    'tqdm',
    'customtkinter.windows.widgets.progressbar',
    'customtkinter.windows.widgets.button',

    # Error handling
    'traceback',
    'logging',# Model training specific imports
    'sklearn.base',
    'sklearn.utils',
    'sklearn.utils.validation',
    'sklearn.exceptions',
    'sklearn.utils.multiclass',
    'sklearn.preprocessing._data',
    'sklearn.ensemble._forest',
    'sklearn.tree',
    'sklearn.tree._tree',
    'sklearn.utils._param_validation',
    'sklearn.model_selection._split',
    'sklearn.utils.fixes',
    'sklearn.metrics._regression',
    'sklearn.metrics._scorer',

    # File handling for model
    'joblib',
    'tempfile',
    'shutil',

    # Threading and process handling
    'threading',
    'queue',
    'multiprocessing',
    'concurrent.futures',

    # Progress tracking
    'tqdm',
    'customtkinter.windows.widgets.progressbar',
    'customtkinter.windows.widgets.button',

    # Error handling
    'traceback',
    'logging',
]

# Collect CustomTkinter dependencies
tmp_ret = collect_all('customtkinter')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Add specific binary dependencies
binaries.extend([
    # Add any specific DLLs or binary files needed for model execution
])


# Collect additional dependencies
for module in ['joblib', 'tqdm']:
    try:
        tmp_ret = collect_all(module)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception as e:
        print(f"Warning: Could not collect all dependencies for {module}: {e}")


# Collect dependencies for major packages
for module in [
    'PIL', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'scipy',
    'geopandas', 'folium', 'skopt', 'dateutil', 'branca', 'geojson',
    'eel', 'typing_extensions', 'bottle', 'Jinja2', 'gevent', 'pyparsing',
    'pyinstaller', 'h11', 'future', 'pillow', 'attrs', 'retry', 'cryptography',
    'setuptools', 'pytz', 'PyYAML', 'cattrs', 'cffi', 'pyaml', 'decorator',
    'threadpoolctl', 'platformdirs', 'joblib', 'packaging', 'xyzservices',
    'greenlet', 'MarkupSafe', 'openpyxl', 'python_dateutil', 'pyproj', 'certifi',
    'pyogrio', 'shapely', 'cycler', 'jh2', 'qh3', 'pywebview', 'proxy_tools',
    'altgraph', 'six', 'idna', 'wassima', 'urllib3', 'et_xmlfile', 'contourpy',
    'fonttools', 'pycparser', 'clr_loader', 'darkdetect', 'kiwisolver',
    'pefile', 'pywin32', 'customtkinter', 'flatbuffers', 'openmeteo_sdk',
    'niquests', 'openmeteo_requests'
]:
    try:
        tmp_ret = collect_all(module)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception as e:
        print(f"Warning: Could not collect all dependencies for {module}: {e}")

a = Analysis(
    ['login.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=exclude_dirs,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BloomSentry',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Icons/AppIcon.png'
)