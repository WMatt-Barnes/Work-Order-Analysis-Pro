"""
Enhanced PM Analysis Module
Provides preventive maintenance frequency analysis, effectiveness metrics,
and optimization recommendations based on work order data, MTBF, and Weibull analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import weibull_min

class PMAnalysis:
    """Enhanced preventive maintenance analysis and optimization"""
    
    def __init__(self):
        self.pm_data = {}
        self.breakdown_data = {}
        self.effectiveness_metrics = {}
        self.weibull_analyzer = None
        
    def set_weibull_analyzer(self, weibull_analyzer):
        """Set Weibull analyzer for integration"""
        self.weibull_analyzer = weibull_analyzer
        
    def separate_pm_and_breakdowns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate PM work orders from breakdown work orders"""
        try:
            if df.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # Common PM work type indicators
            pm_indicators = [
                'pm', 'preventive', 'preventative', 'scheduled', 'routine', 'inspection',
                'lubrication', 'calibration', 'cleaning', 'adjustment', 'testing'
            ]
            
            # Identify PM work orders
            pm_mask = df['Work Type'].str.lower().isin(pm_indicators) | \
                     df['Description'].str.lower().str.contains('|'.join(pm_indicators), na=False)
            
            pm_df = df[pm_mask].copy()
            breakdown_df = df[~pm_mask].copy()
            
            logging.info(f"Separated {len(pm_df)} PM work orders and {len(breakdown_df)} breakdown work orders")
            return pm_df, breakdown_df
            
        except Exception as e:
            logging.error(f"Error separating PM and breakdown data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_mtbf_for_equipment(self, df: pd.DataFrame, equipment_filter: str = None) -> float:
        """Calculate MTBF for specific equipment or all equipment using Crow-AMSAA method"""
        try:
            if df.empty:
                return 0.0
                
            # Filter by equipment if specified
            if equipment_filter:
                filtered_df = df[df['Equipment #'] == equipment_filter]
            else:
                filtered_df = df
                
            if filtered_df.empty:
                return 0.0
            
            # Use Crow-AMSAA method for consistent MTBF calculation
            # Parse dates and sort
            dates = pd.to_datetime(filtered_df['Reported Date'], errors='coerce').dropna()
            dates = sorted(dates)
            
            if len(dates) < 2:
                return 0.0
            
            # Calculate Crow-AMSAA parameters manually
            try:
                from datetime import datetime
                import numpy as np
                
                # Calculate failure times from start
                t0 = dates[0]
                times = [(d - t0).days + 1 for d in dates]
                n = np.arange(1, len(times) + 1)
                
                # Fit Crow-AMSAA model
                log_n = np.log(n)
                log_t = np.log(times)
                coeffs = np.polyfit(log_t, log_n, 1)
                beta = coeffs[0]
                lambda_param = np.exp(coeffs[1])
                
                # Calculate failures per year
                failures_per_year = lambda_param * (365 ** beta)
                
                if failures_per_year > 0:
                    # Convert failures per year to MTBF in days
                    mtbf_days = 365.0 / failures_per_year
                    logging.debug(f"PM MTBF calculated via Crow-AMSAA: {mtbf_days:.2f} days (failures/year: {failures_per_year:.2f})")
                    return round(mtbf_days, 2)
                else:
                    logging.debug("Insufficient data for Crow-AMSAA MTBF calculation in PM analysis")
                    return 0.0
                    
            except Exception as e:
                # Fallback to simple calculation if Crow-AMSAA fails
                logging.warning(f"Could not calculate Crow-AMSAA MTBF: {e}, using simple calculation")
                dates = pd.to_datetime(filtered_df['Reported Date'], errors='coerce').dropna()
                dates = sorted(dates)
                
                if len(dates) < 2:
                    return 0.0
                    
                # Calculate time differences in days
                time_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                mtbf = sum(time_diffs) / len(time_diffs) if time_diffs else 0.0
                
                return round(mtbf, 2)
            
        except Exception as e:
            logging.error(f"Error calculating MTBF: {e}")
            return 0.0
    
    def get_weibull_parameters(self, df: pd.DataFrame, equipment_filter: str = None) -> Dict[str, Any]:
        """Get Weibull parameters for equipment failure analysis"""
        try:
            if not self.weibull_analyzer:
                return {"beta": 1.0, "eta": 365.0, "failure_type": "Random", "confidence": "Low"}
                
            # Filter by equipment if specified
            if equipment_filter:
                filtered_df = df[df['Equipment #'] == equipment_filter]
            else:
                filtered_df = df
                
            if filtered_df.empty:
                return {"beta": 1.0, "eta": 365.0, "failure_type": "Random", "confidence": "Low"}
                
            # Calculate failure times
            failure_times = self.weibull_analyzer.calculate_failure_times(filtered_df)
            
            if len(failure_times) < 2:
                return {"beta": 1.0, "eta": 365.0, "failure_type": "Random", "confidence": "Low"}
                
            # Get Weibull parameters
            beta, eta = self.weibull_analyzer.weibull_mle(failure_times)
            
            # Determine failure type based on beta
            if beta < 1.0:
                failure_type = "Infant Mortality"
                confidence = "High" if len(failure_times) >= 5 else "Medium"
            elif beta < 1.5:
                failure_type = "Random"
                confidence = "High" if len(failure_times) >= 5 else "Medium"
            elif beta < 3.0:
                failure_type = "Wear Out"
                confidence = "High" if len(failure_times) >= 5 else "Medium"
            else:
                failure_type = "Severe Wear Out"
                confidence = "High" if len(failure_times) >= 5 else "Medium"
                
            return {
                "beta": round(beta, 3),
                "eta": round(eta, 1),
                "failure_type": failure_type,
                "confidence": confidence,
                "failure_times": failure_times
            }
            
        except Exception as e:
            logging.error(f"Error getting Weibull parameters: {e}")
            return {"beta": 1.0, "eta": 365.0, "failure_type": "Random", "confidence": "Low"}
    
    def calculate_optimal_pm_frequency(self, mtbf: float, weibull_params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal PM frequency based on MTBF and Weibull parameters"""
        try:
            beta = weibull_params.get("beta", 1.0)
            eta = weibull_params.get("eta", 365.0)
            failure_type = weibull_params.get("failure_type", "Random")
            
            # Base PM interval calculation
            if mtbf > 0:
                # For wear-out failures (beta > 1), PM should be more frequent than MTBF
                if beta > 1.5:
                    # Wear-out: PM at 60-80% of MTBF
                    pm_interval_mtbf = mtbf * 0.7
                elif beta > 1.0:
                    # Early wear-out: PM at 70-90% of MTBF
                    pm_interval_mtbf = mtbf * 0.8
                else:
                    # Random or infant mortality: PM at 80-120% of MTBF
                    pm_interval_mtbf = mtbf * 1.0
            else:
                pm_interval_mtbf = 365.0  # Default to annual
            
            # Weibull-based PM interval calculation
            if beta > 1.0:
                # For wear-out failures, use characteristic life (eta) as reference
                # PM should be scheduled before the failure rate starts increasing significantly
                pm_interval_weibull = eta * (0.3 ** (1/beta))  # 30% of characteristic life
            else:
                # For random failures, use eta as reference
                pm_interval_weibull = eta * 0.5  # 50% of characteristic life
            
            # Combine both approaches with weighting
            if mtbf > 0 and eta > 0:
                # Weight based on data quality
                mtbf_weight = 0.6 if mtbf > 30 else 0.3  # Prefer MTBF if it's reasonable
                weibull_weight = 1.0 - mtbf_weight
                
                optimal_interval = (pm_interval_mtbf * mtbf_weight + 
                                  pm_interval_weibull * weibull_weight)
            elif mtbf > 0:
                optimal_interval = pm_interval_mtbf
            elif eta > 0:
                optimal_interval = pm_interval_weibull
            else:
                optimal_interval = 365.0  # Default annual PM
            
            # Convert to practical intervals
            if optimal_interval < 30:
                practical_interval = "Weekly"
                interval_days = 7
            elif optimal_interval < 90:
                practical_interval = "Monthly"
                interval_days = 30
            elif optimal_interval < 180:
                practical_interval = "Quarterly"
                interval_days = 90
            elif optimal_interval < 365:
                practical_interval = "Semi-annual"
                interval_days = 180
            else:
                practical_interval = "Annual"
                interval_days = 365
            
            # Calculate PM frequency per year
            pm_frequency_per_year = 365 / interval_days
            
            return {
                "optimal_interval_days": round(optimal_interval, 1),
                "practical_interval": practical_interval,
                "interval_days": interval_days,
                "pm_frequency_per_year": round(pm_frequency_per_year, 1),
                "mtbf_based_interval": round(pm_interval_mtbf, 1),
                "weibull_based_interval": round(pm_interval_weibull, 1),
                "failure_type": failure_type,
                "recommendation_basis": f"Based on {failure_type} failure pattern (β={beta}) and MTBF={mtbf} days"
            }
            
        except Exception as e:
            logging.error(f"Error calculating optimal PM frequency: {e}")
            return {
                "optimal_interval_days": 365.0,
                "practical_interval": "Annual",
                "interval_days": 365,
                "pm_frequency_per_year": 1.0,
                "mtbf_based_interval": 365.0,
                "weibull_based_interval": 365.0,
                "failure_type": "Unknown",
                "recommendation_basis": "Default recommendation due to insufficient data"
            }
    
    def calculate_pm_effectiveness(self, pm_df: pd.DataFrame, breakdown_df: pd.DataFrame, 
                                 equipment_filter: str = None) -> Dict[str, Any]:
        """Calculate enhanced PM effectiveness metrics with MTBF and Weibull analysis"""
        try:
            if pm_df.empty and breakdown_df.empty:
                return {"status": "No data available"}
            
            # Combine data for analysis
            combined_df = pd.concat([pm_df, breakdown_df], ignore_index=True)
            
            # Calculate MTBF
            mtbf = self.calculate_mtbf_for_equipment(combined_df, equipment_filter)
            
            # Get Weibull parameters
            weibull_params = self.get_weibull_parameters(combined_df, equipment_filter)
            
            # Calculate optimal PM frequency
            pm_frequency_analysis = self.calculate_optimal_pm_frequency(mtbf, weibull_params)
            
            # Filter by equipment if specified
            if equipment_filter:
                pm_df = pm_df[pm_df['Equipment #'] == equipment_filter]
                breakdown_df = breakdown_df[breakdown_df['Equipment #'] == equipment_filter]
            
            # Calculate basic metrics
            total_pm = len(pm_df)
            total_breakdowns = len(breakdown_df)
            total_work_orders = total_pm + total_breakdowns
            
            # Calculate costs
            pm_cost = pm_df['Work Order Cost'].sum() if 'Work Order Cost' in pm_df.columns else 0
            breakdown_cost = breakdown_df['Work Order Cost'].sum() if 'Work Order Cost' in breakdown_df.columns else 0
            total_cost = pm_cost + breakdown_cost
            
            # Calculate PM ratio
            pm_ratio = total_pm / total_work_orders if total_work_orders > 0 else 0
            
            # Calculate cost effectiveness
            pm_cost_ratio = pm_cost / total_cost if total_cost > 0 else 0
            breakdown_cost_ratio = breakdown_cost / total_cost if total_cost > 0 else 0
            
            # Calculate time-based metrics
            pm_frequency = self._calculate_pm_frequency(pm_df)
            breakdown_frequency = self._calculate_breakdown_frequency(breakdown_df)
            
            # Calculate PM effectiveness score
            effectiveness_score = self._calculate_effectiveness_score(
                pm_ratio, pm_cost_ratio, breakdown_frequency
            )
            
            # Generate enhanced recommendations
            recommendations = self._generate_enhanced_recommendations(
                pm_ratio, pm_cost_ratio, breakdown_frequency, effectiveness_score,
                mtbf, weibull_params, pm_frequency_analysis
            )
            
            return {
                "status": "Analysis complete",
                "total_pm": total_pm,
                "total_breakdowns": total_breakdowns,
                "total_work_orders": total_work_orders,
                "pm_ratio": pm_ratio,
                "pm_cost": pm_cost,
                "breakdown_cost": breakdown_cost,
                "total_cost": total_cost,
                "pm_cost_ratio": pm_cost_ratio,
                "breakdown_cost_ratio": breakdown_cost_ratio,
                "pm_frequency": pm_frequency,
                "breakdown_frequency": breakdown_frequency,
                "effectiveness_score": effectiveness_score,
                "mtbf_days": mtbf,
                "weibull_params": weibull_params,
                "pm_frequency_analysis": pm_frequency_analysis,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logging.error(f"Error calculating PM effectiveness: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def _calculate_pm_frequency(self, pm_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate PM frequency metrics"""
        try:
            if pm_df.empty:
                return {"pm_per_year": 0.0, "avg_pm_interval_days": 0.0}
            
            # Calculate PM frequency per year
            if 'Reported Date' in pm_df.columns:
                dates = pd.to_datetime(pm_df['Reported Date'], errors='coerce').dropna()
                if len(dates) > 1:
                    date_range = (dates.max() - dates.min()).days
                    pm_per_year = (len(dates) / date_range) * 365 if date_range > 0 else 0
                    avg_interval = date_range / len(dates) if len(dates) > 1 else 0
                else:
                    pm_per_year = 0.0
                    avg_interval = 0.0
            else:
                pm_per_year = 0.0
                avg_interval = 0.0
            
            return {
                "pm_per_year": pm_per_year,
                "avg_pm_interval_days": avg_interval
            }
            
        except Exception as e:
            logging.error(f"Error calculating PM frequency: {e}")
            return {"pm_per_year": 0.0, "avg_pm_interval_days": 0.0}
    
    def _calculate_breakdown_frequency(self, breakdown_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate breakdown frequency metrics"""
        try:
            if breakdown_df.empty:
                return {"breakdowns_per_year": 0.0, "avg_time_between_breakdowns": 0.0}
            
            # Calculate breakdown frequency per year
            if 'Reported Date' in breakdown_df.columns:
                dates = pd.to_datetime(breakdown_df['Reported Date'], errors='coerce').dropna()
                if len(dates) > 1:
                    date_range = (dates.max() - dates.min()).days
                    breakdowns_per_year = (len(dates) / date_range) * 365 if date_range > 0 else 0
                    avg_time_between = date_range / len(dates) if len(dates) > 1 else 0
                else:
                    breakdowns_per_year = 0.0
                    avg_time_between = 0.0
            else:
                breakdowns_per_year = 0.0
                avg_time_between = 0.0
            
            return {
                "breakdowns_per_year": breakdowns_per_year,
                "avg_time_between_breakdowns": avg_time_between
            }
            
        except Exception as e:
            logging.error(f"Error calculating breakdown frequency: {e}")
            return {"breakdowns_per_year": 0.0, "avg_time_between_breakdowns": 0.0}
    
    def _calculate_effectiveness_score(self, pm_ratio: float, pm_cost_ratio: float, 
                                     breakdown_frequency: Dict[str, float]) -> float:
        """Calculate overall PM effectiveness score (0-100)"""
        try:
            # Weight factors for different metrics
            pm_ratio_weight = 0.3
            pm_cost_weight = 0.3
            breakdown_freq_weight = 0.4
            
            # PM ratio score (0-100)
            pm_ratio_score = min(pm_ratio * 100, 100)
            
            # PM cost ratio score (0-100) - lower is better for PM cost
            pm_cost_score = max(0, 100 - (pm_cost_ratio * 100))
            
            # Breakdown frequency score (0-100) - lower breakdowns is better
            breakdowns_per_year = breakdown_frequency.get("breakdowns_per_year", 0)
            if breakdowns_per_year <= 1:
                breakdown_score = 100
            elif breakdowns_per_year <= 5:
                breakdown_score = 80
            elif breakdowns_per_year <= 10:
                breakdown_score = 60
            elif breakdowns_per_year <= 20:
                breakdown_score = 40
            else:
                breakdown_score = 20
            
            # Calculate weighted score
            effectiveness_score = (
                pm_ratio_score * pm_ratio_weight +
                pm_cost_score * pm_cost_weight +
                breakdown_score * breakdown_freq_weight
            )
            
            return round(effectiveness_score, 1)
            
        except Exception as e:
            logging.error(f"Error calculating effectiveness score: {e}")
            return 0.0
    
    def _generate_enhanced_recommendations(self, pm_ratio: float, pm_cost_ratio: float,
                                         breakdown_frequency: Dict[str, float], 
                                         effectiveness_score: float,
                                         mtbf: float, weibull_params: Dict[str, Any],
                                         pm_frequency_analysis: Dict[str, Any]) -> List[str]:
        """Generate enhanced PM optimization recommendations"""
        recommendations = []
        
        try:
            # MTBF-based recommendations
            if mtbf > 0:
                recommendations.append(f"MTBF: {mtbf} days - indicates equipment reliability")
                if mtbf < 30:
                    recommendations.append("Low MTBF suggests frequent failures - increase PM frequency")
                elif mtbf > 365:
                    recommendations.append("High MTBF suggests good reliability - consider optimizing PM schedule")
            
            # Weibull-based recommendations
            beta = weibull_params.get("beta", 1.0)
            eta = weibull_params.get("eta", 365.0)
            failure_type = weibull_params.get("failure_type", "Random")
            
            recommendations.append(f"Failure Pattern: {failure_type} (β={beta}, η={eta} days)")
            
            if failure_type == "Wear Out" or failure_type == "Severe Wear Out":
                recommendations.append("Wear-out failure pattern detected - PM is critical to prevent failures")
                recommendations.append("Schedule PM before characteristic life (η) to prevent wear-out failures")
            elif failure_type == "Infant Mortality":
                recommendations.append("Infant mortality pattern - focus on quality control and burn-in testing")
                recommendations.append("PM may be less effective for infant mortality failures")
            else:  # Random
                recommendations.append("Random failure pattern - PM may not prevent failures but can detect issues")
            
            # PM frequency recommendations
            optimal_interval = pm_frequency_analysis.get("optimal_interval_days", 365)
            practical_interval = pm_frequency_analysis.get("practical_interval", "Annual")
            
            recommendations.append(f"Recommended PM Interval: {practical_interval} ({optimal_interval} days)")
            
            # Compare current vs recommended
            current_pm_freq = breakdown_frequency.get("avg_time_between_breakdowns", 0)
            if current_pm_freq > 0 and optimal_interval > 0:
                if current_pm_freq < optimal_interval * 0.5:
                    recommendations.append("Current PM frequency may be too high - consider reducing")
                elif current_pm_freq > optimal_interval * 2:
                    recommendations.append("Current PM frequency may be too low - consider increasing")
            
            # Traditional recommendations
            if pm_ratio < 0.2:
                recommendations.append("Increase PM frequency - current PM ratio is low")
            elif pm_ratio > 0.8:
                recommendations.append("Review PM frequency - may be over-maintaining")
            
            if pm_cost_ratio > 0.7:
                recommendations.append("Optimize PM costs - PM costs are high relative to total")
            elif pm_cost_ratio < 0.2:
                recommendations.append("Consider increasing PM investment - low PM cost ratio")
            
            breakdowns_per_year = breakdown_frequency.get("breakdowns_per_year", 0)
            if breakdowns_per_year > 10:
                recommendations.append("High breakdown frequency - increase PM frequency or improve PM quality")
            elif breakdowns_per_year < 1:
                recommendations.append("Low breakdown frequency - consider optimizing PM schedule")
            
            if effectiveness_score < 50:
                recommendations.append("Overall PM effectiveness is poor - comprehensive review needed")
            elif effectiveness_score < 70:
                recommendations.append("PM effectiveness needs improvement - focus on high-impact areas")
            elif effectiveness_score >= 80:
                recommendations.append("Good PM effectiveness - maintain current practices")
            
            if not recommendations:
                recommendations.append("PM program appears balanced - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def create_pm_analysis_plot(self, pm_df: pd.DataFrame, breakdown_df: pd.DataFrame, 
                               frame: ttk.Frame, title: str = "PM Analysis") -> plt.Figure:
        """Create PM analysis visualization"""
        try:
            if pm_df.empty and breakdown_df.empty:
                # Create empty plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No PM or breakdown data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                return fig
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: Work Order Distribution
            work_order_types = ['PM', 'Breakdown']
            work_order_counts = [len(pm_df), len(breakdown_df)]
            colors = ['lightblue', 'lightcoral']
            
            ax1.pie(work_order_counts, labels=work_order_types, colors=colors, autopct='%1.1f%%')
            ax1.set_title('Work Order Distribution')
            
            # Plot 2: Cost Distribution
            pm_cost = pm_df['Work Order Cost'].sum() if 'Work Order Cost' in pm_df.columns else 0
            breakdown_cost = breakdown_df['Work Order Cost'].sum() if 'Work Order Cost' in breakdown_df.columns else 0
            
            cost_types = ['PM Cost', 'Breakdown Cost']
            cost_values = [pm_cost, breakdown_cost]
            
            bars = ax2.bar(cost_types, cost_values, color=colors)
            ax2.set_title('Cost Distribution')
            ax2.set_ylabel('Cost ($)')
            
            # Add value labels on bars
            for bar, value in zip(bars, cost_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'${value:,.0f}', ha='center', va='bottom')
            
            # Plot 3: Monthly Trend
            if 'Reported Date' in pm_df.columns or 'Reported Date' in breakdown_df.columns:
                self._plot_monthly_trend(ax3, pm_df, breakdown_df)
            else:
                ax3.text(0.5, 0.5, "No date data available", ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Monthly Trend')
            
            # Plot 4: Equipment Analysis
            self._plot_equipment_analysis(ax4, pm_df, breakdown_df)
            
            plt.tight_layout()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating PM analysis plot: {e}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error creating PM analysis plot: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(title)
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return fig
    
    def _plot_monthly_trend(self, ax, pm_df: pd.DataFrame, breakdown_df: pd.DataFrame):
        """Plot monthly trend of PM vs breakdown work orders"""
        try:
            # Combine data and create monthly trend
            all_data = []
            
            if not pm_df.empty and 'Reported Date' in pm_df.columns:
                pm_dates = pd.to_datetime(pm_df['Reported Date'], errors='coerce').dropna()
                for date in pm_dates:
                    all_data.append({'date': date, 'type': 'PM'})
            
            if not breakdown_df.empty and 'Reported Date' in breakdown_df.columns:
                breakdown_dates = pd.to_datetime(breakdown_df['Reported Date'], errors='coerce').dropna()
                for date in breakdown_dates:
                    all_data.append({'date': date, 'type': 'Breakdown'})
            
            if all_data:
                df_trend = pd.DataFrame(all_data)
                df_trend['month'] = df_trend['date'].dt.to_period('M')
                
                monthly_counts = df_trend.groupby(['month', 'type']).size().unstack(fill_value=0)
                
                if not monthly_counts.empty:
                    monthly_counts.plot(kind='line', ax=ax, marker='o')
                    ax.set_title('Monthly Work Order Trend')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Number of Work Orders')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, "No trend data available", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Monthly Trend')
            else:
                ax.text(0.5, 0.5, "No date data available", ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Monthly Trend')
                
        except Exception as e:
            logging.error(f"Error plotting monthly trend: {e}")
            ax.text(0.5, 0.5, "Error plotting trend", ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Trend')
    
    def _plot_equipment_analysis(self, ax, pm_df: pd.DataFrame, breakdown_df: pd.DataFrame):
        """Plot equipment-level PM vs breakdown analysis"""
        try:
            # Get equipment with both PM and breakdown data
            pm_equipment = set(pm_df['Equipment #'].unique()) if 'Equipment #' in pm_df.columns else set()
            breakdown_equipment = set(breakdown_df['Equipment #'].unique()) if 'Equipment #' in breakdown_df.columns else set()
            all_equipment = pm_equipment.union(breakdown_equipment)
            
            if len(all_equipment) > 0:
                equipment_data = []
                
                for equipment in list(all_equipment)[:10]:  # Limit to top 10
                    pm_count = len(pm_df[pm_df['Equipment #'] == equipment]) if equipment in pm_equipment else 0
                    breakdown_count = len(breakdown_df[breakdown_df['Equipment #'] == equipment]) if equipment in breakdown_equipment else 0
                    
                    equipment_data.append({
                        'equipment': str(equipment),
                        'pm_count': pm_count,
                        'breakdown_count': breakdown_count
                    })
                
                if equipment_data:
                    df_equipment = pd.DataFrame(equipment_data)
                    df_equipment = df_equipment.sort_values('breakdown_count', ascending=False)
                    
                    x = np.arange(len(df_equipment))
                    width = 0.35
                    
                    ax.bar(x - width/2, df_equipment['pm_count'], width, label='PM', color='lightblue')
                    ax.bar(x + width/2, df_equipment['breakdown_count'], width, label='Breakdown', color='lightcoral')
                    
                    ax.set_title('Equipment Analysis (Top 10)')
                    ax.set_xlabel('Equipment')
                    ax.set_ylabel('Work Order Count')
                    ax.set_xticks(x)
                    ax.set_xticklabels(df_equipment['equipment'], rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, "No equipment data available", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Equipment Analysis')
            else:
                ax.text(0.5, 0.5, "No equipment data available", ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Equipment Analysis')
                
        except Exception as e:
            logging.error(f"Error plotting equipment analysis: {e}")
            ax.text(0.5, 0.5, "Error plotting equipment analysis", ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Equipment Analysis')
    
    def optimize_pm_schedule(self, pm_df: pd.DataFrame, breakdown_df: pd.DataFrame,
                           equipment_filter: str = None) -> Dict[str, Any]:
        """Generate PM schedule optimization recommendations"""
        try:
            if pm_df.empty and breakdown_df.empty:
                return {"status": "No data available for optimization"}
            
            # Filter by equipment if specified
            if equipment_filter:
                pm_df = pm_df[pm_df['Equipment #'] == equipment_filter]
                breakdown_df = breakdown_df[breakdown_df['Equipment #'] == equipment_filter]
            
            # Calculate current metrics
            current_metrics = self.calculate_pm_effectiveness(pm_df, breakdown_df, equipment_filter)
            
            # Generate optimization recommendations
            optimization_recommendations = []
            
            # PM frequency optimization
            pm_freq = current_metrics.get("pm_frequency", {})
            breakdown_freq = current_metrics.get("breakdown_frequency", {})
            
            pm_per_year = pm_freq.get("pm_per_year", 0)
            breakdowns_per_year = breakdown_freq.get("breakdowns_per_year", 0)
            
            if breakdowns_per_year > pm_per_year * 2:
                optimization_recommendations.append({
                    "type": "Increase PM Frequency",
                    "current": f"{pm_per_year:.1f} PM/year",
                    "recommended": f"{breakdowns_per_year * 0.8:.1f} PM/year",
                    "rationale": "High breakdown frequency suggests insufficient PM"
                })
            elif breakdowns_per_year < pm_per_year * 0.5:
                optimization_recommendations.append({
                    "type": "Decrease PM Frequency",
                    "current": f"{pm_per_year:.1f} PM/year",
                    "recommended": f"{pm_per_year * 0.7:.1f} PM/year",
                    "rationale": "Low breakdown frequency suggests over-maintenance"
                })
            
            # Cost optimization
            pm_cost_ratio = current_metrics.get("pm_cost_ratio", 0)
            if pm_cost_ratio > 0.7:
                optimization_recommendations.append({
                    "type": "Optimize PM Costs",
                    "current": f"{pm_cost_ratio:.1%} of total cost",
                    "recommended": "Target 40-60% of total cost",
                    "rationale": "PM costs are disproportionately high"
                })
            
            return {
                "status": "Optimization analysis complete",
                "current_metrics": current_metrics,
                "optimization_recommendations": optimization_recommendations,
                "estimated_savings": self._estimate_savings(current_metrics, optimization_recommendations)
            }
            
        except Exception as e:
            logging.error(f"Error optimizing PM schedule: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def _estimate_savings(self, current_metrics: Dict[str, Any], 
                         recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate potential savings from PM optimization"""
        try:
            total_cost = current_metrics.get("total_cost", 0)
            estimated_savings = 0.0
            
            for rec in recommendations:
                if "Decrease PM Frequency" in rec.get("type", ""):
                    # Estimate 20% reduction in PM costs
                    pm_cost = current_metrics.get("pm_cost", 0)
                    estimated_savings += pm_cost * 0.2
                elif "Optimize PM Costs" in rec.get("type", ""):
                    # Estimate 15% reduction in PM costs
                    pm_cost = current_metrics.get("pm_cost", 0)
                    estimated_savings += pm_cost * 0.15
            
            return {
                "estimated_annual_savings": estimated_savings,
                "savings_percentage": (estimated_savings / total_cost * 100) if total_cost > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Error estimating savings: {e}")
            return {"estimated_annual_savings": 0.0, "savings_percentage": 0.0} 