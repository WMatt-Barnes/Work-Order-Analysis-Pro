Work Order Analysis Pro 1.1

A comprehensive reliability engineering software for analyzing maintenance work order data using advanced statistical methods.

## Overview

WOAPro transforms raw maintenance work order data into actionable reliability insights through:

- **Weibull Analysis**: Failure time distribution modeling and reliability predictions
- **Crow-AMSAA Analysis**: Reliability growth modeling and trend analysis
- **Preventive Maintenance Optimization**: PM frequency recommendations and effectiveness analysis
- **Spares Analysis**: Demand forecasting and inventory optimization using Monte Carlo simulation
- **FMEA Export**: Risk priority number calculations and failure mode analysis
- **AI Classification**: Intelligent failure mode classification using NLP and machine learning

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (primary platform)
- 4GB RAM minimum, 8GB recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WMatt-Barnes/Work-Order-Analysis-Pro-V1.1.git
   cd Work-Order-Analysis-Pro-V1.1
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python src/main.py
   ```

### Alternative: Download Release

1. Go to [Releases](https://github.com/WMatt-Barnes/Work-Order-Analysis-Pro-V1.1/releases)
2. Download the latest `WOAPro-v1.1.zip`
3. Extract and run `main.py`

## Basic Navigation

### Main Interface

The application features a tabbed interface with the following sections:

- **Data Input**: Load work order files and configure column mappings
- **Analysis**: Core reliability analysis with Crow-AMSAA plots
- **Risk Assessment**: Multi-dimensional risk scoring and visualization
- **Weibull Analysis**: Detailed failure time distribution analysis
- **PM Analysis**: Preventive maintenance optimization
- **Spares Analysis**: Inventory forecasting and optimization
- **FMEA Export**: Failure mode analysis and risk prioritization

### Key Features

- **Interactive Plots**: Click on data points to view work order details
- **Filtering**: Filter by equipment, failure codes, dates, and work types
- **Export Capabilities**: Excel reports, plots, and FMEA data
- **AI Classification**: Automatic failure mode classification (optional)
- **Batch Processing**: Process multiple files simultaneously

### Data Requirements

**Required Columns:**
- Work Order Number
- Work Description
- Asset Name
- Equipment Number
- Work Type
- Date Reported
- Work Order Cost (optional)
- User Failure Code (optional)

**Supported Formats:**
- Excel (.xlsx, .xls)
- CSV files

### Configuration

1. Copy `app_config_template.json` to `app_config.json`
2. Update the paths in `app_config.json` to point to your data files
3. The application will remember your last used paths

## Documentation

- **[Technical Application Guide](docs/Technical_Application_Guide.md)**: Detailed statistical methods and algorithms
- **[User Guide](docs/User_Guide.md)**: Step-by-step usage instructions
- **[Installation Guide](docs/Installation_Guide.md)**: Detailed setup instructions

## Statistical Methods

WOAPro implements industry-standard reliability engineering methods:

- **Weibull Analysis**: Maximum likelihood estimation for failure time modeling
- **Crow-AMSAA**: Reliability growth analysis for repairable systems
- **Monte Carlo Simulation**: Stochastic modeling for spares forecasting
- **Risk Assessment**: Multi-dimensional risk scoring and prioritization
- **Pareto Analysis**: Cost and frequency-based failure mode prioritization

## Support

- **Issues**: Report bugs and feature requests via [GitHub Issues](https://github.com/WMatt-Barnes/Work-Order-Analysis-Pro-V1.1/issues)
- **Documentation**: See the `docs/` folder for detailed guides
- **Examples**: Sample data files included in the `data/` folder

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**WOAPro v1.1** - Professional reliability engineering made accessible 
