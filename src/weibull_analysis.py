"""
Weibull Analysis Module
Provides Weibull parameter estimation, plotting, and reliability calculations
for failure time data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from typing import Tuple, Optional, List, Dict, Any
from scipy.optimize import minimize
from scipy.stats import weibull_min
import logging
from datetime import datetime, timedelta

class WeibullAnalysis:
    """Weibull analysis for failure time data"""
    
    def __init__(self):
        self.failure_times = []
        self.weibull_params = None
        self.confidence_bounds = None
        
    def calculate_failure_times(self, df: pd.DataFrame, date_column: str = 'Reported Date') -> List[float]:
        """Calculate failure times in days from sorted dates"""
        try:
            # Parse dates and sort
            dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
            dates = sorted(dates)
            
            if len(dates) < 2:
                logging.warning("Insufficient data for Weibull analysis")
                return []
            
            # Calculate time differences in days
            failure_times = []
            start_date = dates[0]
            
            for date in dates:
                time_diff = (date - start_date).days
                if time_diff > 0:  # Skip same-day failures
                    failure_times.append(time_diff)
            
            # Remove duplicates and sort
            failure_times = sorted(list(set(failure_times)))
            
            logging.info(f"Calculated {len(failure_times)} unique failure times")
            return failure_times
            
        except Exception as e:
            logging.error(f"Error calculating failure times: {e}")
            return []
    
    def weibull_mle(self, failure_times: List[float]) -> Tuple[float, float]:
        """Maximum Likelihood Estimation for Weibull parameters"""
        try:
            if len(failure_times) < 2:
                return 1.0, 365.0  # Default values
            
            # Convert to numpy array
            data = np.array(failure_times)
            
            # Define negative log-likelihood function
            def neg_log_likelihood(params):
                beta, eta = params
                if beta <= 0 or eta <= 0:
                    return np.inf
                
                n = len(data)
                log_likelihood = n * np.log(beta) - n * beta * np.log(eta) + (beta - 1) * np.sum(np.log(data)) - np.sum((data / eta) ** beta)
                return -log_likelihood
            
            # Initial guess: beta=1.0, eta=mean of data
            initial_guess = [1.0, np.mean(data)]
            
            # Optimize
            result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B', 
                           bounds=[(0.1, 10.0), (1.0, 10000.0)])
            
            if result.success:
                beta, eta = result.x
                logging.info(f"Weibull MLE: β={beta:.3f}, η={eta:.1f}")
                return beta, eta
            else:
                logging.warning("MLE optimization failed, using default values")
                return 1.0, np.mean(data)
                
        except Exception as e:
            logging.error(f"Error in Weibull MLE: {e}")
            return 1.0, 365.0
    
    def calculate_reliability(self, time: float, beta: float, eta: float) -> float:
        """Calculate reliability at given time"""
        try:
            return np.exp(-(time / eta) ** beta)
        except Exception as e:
            logging.error(f"Error calculating reliability: {e}")
            return 0.0
    
    def calculate_hazard_rate(self, time: float, beta: float, eta: float) -> float:
        """Calculate hazard rate at given time"""
        try:
            return (beta / eta) * (time / eta) ** (beta - 1)
        except Exception as e:
            logging.error(f"Error calculating hazard rate: {e}")
            return 0.0
    
    def calculate_confidence_bounds(self, failure_times: List[float], confidence: float = 0.95) -> Dict[str, List[float]]:
        """Calculate confidence bounds for Weibull plot"""
        try:
            if len(failure_times) < 3:
                return {"lower": [], "upper": []}
            
            # Simple confidence bounds using normal approximation
            n = len(failure_times)
            z_alpha = 1.96  # 95% confidence
            
            # Calculate median ranks
            median_ranks = [(i - 0.3) / (n + 0.4) for i in range(1, n + 1)]
            
            # Calculate confidence bounds
            lower_bounds = []
            upper_bounds = []
            
            for i, rank in enumerate(median_ranks):
                # Simple approximation for confidence bounds
                se = np.sqrt(rank * (1 - rank) / n)
                lower = max(0.001, rank - z_alpha * se)
                upper = min(0.999, rank + z_alpha * se)
                
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            
            return {"lower": lower_bounds, "upper": upper_bounds}
            
        except Exception as e:
            logging.error(f"Error calculating confidence bounds: {e}")
            return {"lower": [], "upper": []}
    
    def assess_goodness_of_fit(self, failure_times: List[float], beta: float, eta: float) -> Dict[str, Any]:
        """Assess goodness of fit for Weibull distribution"""
        try:
            if len(failure_times) < 3:
                return {
                    "overall_assessment": "Insufficient data",
                    "r_squared": 0.0,
                    "kolmogorov_smirnov": 1.0,
                    "anderson_darling": float('inf'),
                    "fit_quality": "Poor"
                }
            
            # Calculate theoretical CDF values
            theoretical_cdf = [1 - np.exp(-(t / eta) ** beta) for t in failure_times]
            
            # Calculate empirical CDF (median ranks)
            n = len(failure_times)
            empirical_cdf = [(i - 0.3) / (n + 0.4) for i in range(1, n + 1)]
            
            # Calculate R-squared
            mean_empirical = np.mean(empirical_cdf)
            ss_tot = np.sum((np.array(empirical_cdf) - mean_empirical) ** 2)
            ss_res = np.sum((np.array(empirical_cdf) - np.array(theoretical_cdf)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate Kolmogorov-Smirnov statistic
            ks_stat = max(abs(np.array(empirical_cdf) - np.array(theoretical_cdf)))
            
            # Calculate Anderson-Darling statistic (simplified)
            ad_stat = np.sum(((np.array(empirical_cdf) - np.array(theoretical_cdf)) ** 2) / 
                           (np.array(theoretical_cdf) * (1 - np.array(theoretical_cdf))))
            
            # Determine overall fit quality
            if r_squared >= 0.95 and ks_stat <= 0.1:
                fit_quality = "Excellent"
                overall_assessment = "Good fit"
            elif r_squared >= 0.90 and ks_stat <= 0.15:
                fit_quality = "Good"
                overall_assessment = "Good fit"
            elif r_squared >= 0.80 and ks_stat <= 0.20:
                fit_quality = "Fair"
                overall_assessment = "Fair fit"
            else:
                fit_quality = "Poor"
                overall_assessment = "Bad fit"
            
            return {
                "overall_assessment": overall_assessment,
                "r_squared": r_squared,
                "kolmogorov_smirnov": ks_stat,
                "anderson_darling": ad_stat,
                "fit_quality": fit_quality,
                "theoretical_cdf": theoretical_cdf,
                "empirical_cdf": empirical_cdf
            }
            
        except Exception as e:
            logging.error(f"Error assessing goodness of fit: {e}")
            return {
                "overall_assessment": "Error in calculation",
                "r_squared": 0.0,
                "kolmogorov_smirnov": 1.0,
                "anderson_darling": float('inf'),
                "fit_quality": "Error"
            }
    
    def create_weibull_plot(self, failure_times: List[float], frame: ttk.Frame, 
                           title: str = "Weibull Analysis", 
                           start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> Tuple[plt.Figure, float, float]:
        """Create Weibull probability plot with improved layout and dynamic scaling"""
        try:
            logging.info(f"Creating Weibull plot with {len(failure_times)} failure times")
            
            if len(failure_times) < 2:
                logging.warning("Insufficient data for Weibull analysis")
                # Create empty plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "Insufficient data for Weibull analysis", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                return fig, 1.0, 365.0
            
            # Get frame dimensions for dynamic scaling
            frame.update_idletasks()
            width_px = max(frame.winfo_width(), 400)
            height_px = max(frame.winfo_height(), 300)
            width_in = min(max(width_px / 100, 6), 12)
            height_in = min(max(height_px / 100, 4), 8)
            font_scale = width_in / 8
            
            # Calculate Weibull parameters
            beta, eta = self.weibull_mle(failure_times)
            
            # Calculate median ranks for plotting
            n = len(failure_times)
            median_ranks = [(i - 0.3) / (n + 0.4) for i in range(1, n + 1)]
            
            # Calculate confidence bounds
            confidence_bounds = self.calculate_confidence_bounds(failure_times)
            
            # Create Weibull plot with dynamic sizing
            fig, ax = plt.subplots(figsize=(width_in, height_in))
            
            # Plot data points
            ax.scatter(failure_times, median_ranks, color='blue', s=60, label='Observed Failures', zorder=5, alpha=0.7)
            
            # Plot confidence bounds if available
            if confidence_bounds["lower"] and confidence_bounds["upper"]:
                ax.fill_between(failure_times, confidence_bounds["lower"], confidence_bounds["upper"], 
                              alpha=0.3, color='gray', label=f'{int(95)}% Confidence Bounds')
            
            # Plot fitted line
            x_fit = np.linspace(min(failure_times), max(failure_times), 100)
            y_fit = 1 - np.exp(-(x_fit / eta) ** beta)
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Weibull Fit (β={beta:.2f}, η={eta:.0f})')
            
            # Customize plot with dynamic font sizing
            ax.set_xscale('log')
            ax.set_yscale('logit')
            ax.set_xlabel('Time (days)', fontsize=10 * font_scale)
            ax.set_ylabel('Cumulative Probability', fontsize=10 * font_scale)
            
            # Create title with date range if available
            title_text = f'{title}\nβ={beta:.3f}, η={eta:.1f} days'
            if start_date and end_date:
                title_text += f'\nAnalysis Period: {start_date.strftime("%m/%d/%Y")} to {end_date.strftime("%m/%d/%Y")}'
            elif start_date:
                title_text += f'\nFrom: {start_date.strftime("%m/%d/%Y")}'
            elif end_date:
                title_text += f'\nTo: {end_date.strftime("%m/%d/%Y")}'
            
            ax.set_title(title_text, fontsize=12 * font_scale, pad=20)
            ax.grid(True, which="both", ls="--", alpha=0.7)
            ax.legend(fontsize=9 * font_scale, loc='upper left')
            
            # Adjust tick label sizes
            ax.tick_params(axis='both', labelsize=8 * font_scale)
            
            # Add reliability information with better positioning
            reliability_6m = self.calculate_reliability(180, beta, eta)
            reliability_1y = self.calculate_reliability(365, beta, eta)
            reliability_2y = self.calculate_reliability(730, beta, eta)
            reliability_4y = self.calculate_reliability(1460, beta, eta)
            reliability_10y = self.calculate_reliability(3650, beta, eta)
            
            info_text = f'Reliability: 6m={reliability_6m*100:.1f}%, 1y={reliability_1y*100:.1f}%, 2y={reliability_2y*100:.1f}%, 4y={reliability_4y*100:.1f}%, 10y={reliability_10y*100:.1f}%'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8 * font_scale)
            
            # Ensure tight layout to prevent overlap
            plt.tight_layout()
            
            # Create canvas and embed in frame
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            return fig, beta, eta
            
        except Exception as e:
            logging.error(f"Error creating Weibull plot: {e}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error creating Weibull plot: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(title)
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return fig, 1.0, 365.0
    
    def get_analysis_summary(self, failure_times: List[float], 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get comprehensive analysis summary with goodness of fit assessment, including Weibull demand rate (failures/year)"""
        try:
            if len(failure_times) < 2:
                return {
                    "status": "Insufficient data",
                    "beta": 1.0,
                    "eta": 365.0,
                    "reliability_6m": 0.0,
                    "reliability_1y": 0.0,
                    "reliability_2y": 0.0,
                    "reliability_4y": 0.0,
                    "reliability_10y": 0.0,
                    "mean_time_to_failure": 0.0,
                    "failure_count": len(failure_times),
                    "goodness_of_fit": "Insufficient data",
                    "start_date": start_date,
                    "end_date": end_date,
                    "weibull_demand_rate": 0.0
                }
            
            beta, eta = self.weibull_mle(failure_times)
            
            # Calculate reliability at different times (6 Months, 1 Year, 2 Year, 4 Year, 10 Year)
            reliability_6m = self.calculate_reliability(180, beta, eta)  # 6 months = 180 days
            reliability_1y = self.calculate_reliability(365, beta, eta)  # 1 year = 365 days
            reliability_2y = self.calculate_reliability(730, beta, eta)  # 2 years = 730 days
            reliability_4y = self.calculate_reliability(1460, beta, eta)  # 4 years = 1460 days
            reliability_10y = self.calculate_reliability(3650, beta, eta)  # 10 years = 3650 days
            
            # Calculate mean time to failure
            import math
            mttf = eta * math.gamma(1 + 1/beta)
            weibull_demand_rate = 365.0 / mttf if mttf > 0 else 0.0
            
            # Assess goodness of fit
            goodness_of_fit = self.assess_goodness_of_fit(failure_times, beta, eta)
            
            return {
                "status": "Analysis complete",
                "beta": beta,
                "eta": eta,
                "reliability_6m": reliability_6m,
                "reliability_1y": reliability_1y,
                "reliability_2y": reliability_2y,
                "reliability_4y": reliability_4y,
                "reliability_10y": reliability_10y,
                "mean_time_to_failure": mttf,
                "failure_count": len(failure_times),
                "shape_parameter": beta,
                "scale_parameter": eta,
                "goodness_of_fit": goodness_of_fit["overall_assessment"],
                "fit_quality": goodness_of_fit["fit_quality"],
                "r_squared": goodness_of_fit["r_squared"],
                "kolmogorov_smirnov": goodness_of_fit["kolmogorov_smirnov"],
                "start_date": start_date,
                "end_date": end_date,
                "weibull_demand_rate": weibull_demand_rate
            }
            
        except Exception as e:
            logging.error(f"Error getting analysis summary: {e}")
            return {
                "status": f"Error: {str(e)}",
                "beta": 1.0,
                "eta": 365.0,
                "reliability_6m": 0.0,
                "reliability_1y": 0.0,
                "reliability_2y": 0.0,
                "reliability_4y": 0.0,
                "reliability_10y": 0.0,
                "mean_time_to_failure": 0.0,
                "failure_count": len(failure_times) if 'failure_times' in locals() else 0,
                "goodness_of_fit": "Error in calculation",
                "fit_quality": "Error",
                "r_squared": 0.0,
                "kolmogorov_smirnov": 1.0,
                "start_date": start_date,
                "end_date": end_date,
                "weibull_demand_rate": 0.0
            } 