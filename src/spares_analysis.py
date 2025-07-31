"""
Spares Analysis Module
Provides spares demand forecasting, stockout risk assessment, and economic order quantity
calculations using Monte Carlo simulation and statistical analysis.
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
from scipy.stats import poisson, norm, weibull_min
import random

class SparesAnalysis:
    """Spares analysis and optimization"""
    
    def __init__(self):
        self.demand_history = {}
        self.weibull_params = {}
        self.simulation_results = {}
        
    def analyze_spares_demand(self, df: pd.DataFrame, equipment_filter: Optional[str] = None, included_indices: Optional[set] = None) -> Dict[str, Any]:
        """Analyze spares demand patterns from work order data, accounting for censored data"""
        try:
            if df.empty:
                return {"status": "No data available"}
            
            # Filter by equipment if specified
            if equipment_filter:
                filtered_df = df[df['Equipment #'] == equipment_filter].copy()
                df = filtered_df
            
            # Filter by included indices if provided (to account for censored data)
            if included_indices is not None:
                df = df[df.index.isin(included_indices)].copy()
            
            # Group by failure code to identify spares requirements
            failure_analysis = df.groupby(['Failure Code', 'Failure Description']).agg({
                'Work Order': 'count',
                'Equipment #': lambda x: list(x.unique()),
                'Reported Date': ['min', 'max'],
                'Work Order Cost': 'sum'
            }).reset_index()
            
            # Flatten column names
            failure_analysis.columns = ['Failure Code', 'Failure Description', 'Demand Count', 
                                      'Equipment List', 'First Failure', 'Last Failure', 'Total Cost']
            
            # Calculate demand metrics
            spares_demand = {}
            total_demand_rate = 0.0
            
            for _, row in failure_analysis.iterrows():
                failure_code = row['Failure Code']
                
                # Calculate time span
                first_failure = pd.to_datetime(row['First Failure'])
                last_failure = pd.to_datetime(row['Last Failure'])
                demand_count = int(row['Demand Count'])
                time_span_days = (last_failure - first_failure).days
                
                # Debug logging
                logging.debug(f"Failure code: {failure_code}, First failure: {first_failure}, Last failure: {last_failure}, Demand count: {demand_count}, Time span (days): {time_span_days}")
                
                # Calculate demand rate using intervals
                if time_span_days > 0 and demand_count > 1:
                    demand_rate_per_year = ((demand_count - 1) / time_span_days) * 365
                else:
                    demand_rate_per_year = 0
                
                total_demand_rate += demand_rate_per_year
                
                # Calculate average cost per demand
                avg_cost_per_demand = row['Total Cost'] / demand_count if demand_count > 0 else 0
                
                spares_demand[failure_code] = {
                    'failure_code': failure_code,
                    'failure_description': row['Failure Description'],
                    'demand_count': demand_count,
                    'demand_rate_per_year': float(demand_rate_per_year),
                    'time_span_days': int(time_span_days),
                    'total_cost': float(row['Total Cost']),
                    'avg_cost_per_demand': float(avg_cost_per_demand),
                    'equipment_count': len(row['Equipment List']),
                    'equipment_list': row['Equipment List']
                }
            
            # Calculate overall statistics
            total_demands = sum(data['demand_count'] for data in spares_demand.values())
            total_cost = sum(data['total_cost'] for data in spares_demand.values())
            avg_demand_rate = np.mean([data['demand_rate_per_year'] for data in spares_demand.values()]) if spares_demand else 0
            
            return {
                "status": "Analysis complete",
                "spares_demand": spares_demand,
                "total_demands": total_demands,
                "total_cost": total_cost,
                "avg_demand_rate": avg_demand_rate,
                "total_demand_rate": total_demand_rate,  # Sum of all failure mode rates
                "unique_failure_modes": len(spares_demand)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing spares demand: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def monte_carlo_demand_simulation(self, demand_rate: float, simulation_days: int = 365, 
                                    num_simulations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo simulation for demand forecasting"""
        try:
            if demand_rate <= 0:
                return {"status": "Invalid demand rate"}
            
            # Use Poisson distribution for demand simulation
            lambda_param = demand_rate * (simulation_days / 365)  # Convert to simulation period
            
            # Run simulations
            simulation_results = []
            for _ in range(num_simulations):
                # Generate demand for each day
                daily_demands = np.random.poisson(lambda_param / simulation_days, simulation_days)
                cumulative_demand = np.cumsum(daily_demands)
                simulation_results.append(cumulative_demand)
            
            # Calculate statistics
            simulation_array = np.array(simulation_results)
            mean_demand = np.mean(simulation_array[:, -1])  # Final cumulative demand
            std_demand = np.std(simulation_array[:, -1])
            
            # Calculate percentiles
            percentiles = {
                'p10': np.percentile(simulation_array[:, -1], 10),
                'p25': np.percentile(simulation_array[:, -1], 25),
                'p50': np.percentile(simulation_array[:, -1], 50),
                'p75': np.percentile(simulation_array[:, -1], 75),
                'p90': np.percentile(simulation_array[:, -1], 90),
                'p95': np.percentile(simulation_array[:, -1], 95),
                'p99': np.percentile(simulation_array[:, -1], 99)
            }
            
            return {
                "status": "Simulation complete",
                "mean_demand": mean_demand,
                "std_demand": std_demand,
                "percentiles": percentiles,
                "simulation_results": simulation_results,
                "simulation_days": simulation_days,
                "num_simulations": num_simulations
            }
            
        except Exception as e:
            logging.error(f"Error in Monte Carlo simulation: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def monte_carlo_weibull_simulation(self, weibull_params: Dict[str, Any], 
                                     equipment_data: pd.DataFrame,
                                     simulation_years: int = 10,
                                     num_simulations: int = 1000,
                                     lead_time_days: int = 30) -> Dict[str, Any]:
        """Perform Monte Carlo simulation using Weibull parameters for equipment-specific spares analysis, including lead time window stats"""
        try:
            if not weibull_params or 'beta' not in weibull_params or 'eta' not in weibull_params:
                return {"status": "Invalid Weibull parameters"}
            
            beta = weibull_params['beta']
            eta = weibull_params['eta']
            
            # Get simulation start date from equipment data
            if equipment_data.empty:
                return {"status": "No equipment data available"}
            
            # Find the first date in the data
            start_date = pd.to_datetime(equipment_data['Reported Date'].min())
            simulation_days = simulation_years * 365
            
            # Calculate MTBF from Weibull parameters
            import math
            mtbf = eta * math.gamma(1 + 1/beta)
            
            # Run Monte Carlo simulations
            simulation_results = []
            failure_times_simulations = []
            lead_time_failures = []  # Track failures within lead time window
            
            for sim in range(num_simulations):
                # Generate failure times using Weibull distribution
                failure_times = []
                current_time = 0
                
                while current_time < simulation_days:
                    # Generate time to next failure using Weibull distribution
                    time_to_failure = weibull_min.rvs(beta, scale=eta)
                    current_time += time_to_failure
                    
                    if current_time <= simulation_days:
                        failure_times.append(current_time)
                
                # Calculate cumulative failures over time
                daily_failures = np.zeros(simulation_days)
                for failure_time in failure_times:
                    day_index = int(failure_time)
                    if day_index < simulation_days:
                        daily_failures[day_index] += 1
                
                cumulative_failures = np.cumsum(daily_failures)
                simulation_results.append(cumulative_failures)
                failure_times_simulations.append(failure_times)
                # Count failures within lead time window
                lead_time_failures.append(np.sum(np.array(failure_times) <= lead_time_days))
            
            # Calculate statistics
            simulation_array = np.array(simulation_results)
            mean_failures = np.mean(simulation_array[:, -1])  # Total failures at end
            std_failures = np.std(simulation_array[:, -1])
            
            # Calculate percentiles
            percentiles = {
                'p10': np.percentile(simulation_array[:, -1], 10),
                'p25': np.percentile(simulation_array[:, -1], 25),
                'p50': np.percentile(simulation_array[:, -1], 50),
                'p75': np.percentile(simulation_array[:, -1], 75),
                'p90': np.percentile(simulation_array[:, -1], 90),
                'p95': np.percentile(simulation_array[:, -1], 95),
                'p99': np.percentile(simulation_array[:, -1], 99)
            }
            
            # Lead time window stats
            lead_time_stats = {
                'mean': float(np.mean(lead_time_failures)),
                'std': float(np.std(lead_time_failures)),
                'p10': float(np.percentile(lead_time_failures, 10)),
                'p25': float(np.percentile(lead_time_failures, 25)),
                'p50': float(np.percentile(lead_time_failures, 50)),
                'p75': float(np.percentile(lead_time_failures, 75)),
                'p90': float(np.percentile(lead_time_failures, 90)),
                'p95': float(np.percentile(lead_time_failures, 95)),
                'p99': float(np.percentile(lead_time_failures, 99)),
                'all': lead_time_failures
            }
            
            # Calculate optimal stocking levels
            # For 95% service level
            optimal_stock_95 = int(percentiles['p95'])
            # For 99% service level  
            optimal_stock_99 = int(percentiles['p99'])
            
            # Calculate annual failure rate
            annual_failure_rate = mean_failures / simulation_years
            
            return {
                "status": "Weibull simulation complete",
                "weibull_params": weibull_params,
                "mtbf": mtbf,
                "annual_failure_rate": annual_failure_rate,
                "mean_failures": mean_failures,
                "std_failures": std_failures,
                "percentiles": percentiles,
                "optimal_stock_95": optimal_stock_95,
                "optimal_stock_99": optimal_stock_99,
                "simulation_results": simulation_results,
                "failure_times_simulations": failure_times_simulations,
                "lead_time_stats": lead_time_stats,
                "simulation_years": simulation_years,
                "num_simulations": num_simulations,
                "start_date": start_date,
                "simulation_days": simulation_days
            }
            
        except Exception as e:
            logging.error(f"Error in Weibull Monte Carlo simulation: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def calculate_optimal_stocking_levels(self, weibull_simulation: Dict[str, Any],
                                        mtbf: float,
                                        lead_time_days: int = 30,
                                        service_level: float = 0.95) -> Dict[str, Any]:
        """Calculate optimal stocking levels using both Weibull and MTBF approaches, factoring in lead time window"""
        try:
            if weibull_simulation.get("status") != "Weibull simulation complete":
                return {"status": "Invalid simulation results"}
            
            # Get Weibull-based recommendations (lead time window)
            lead_time_stats = weibull_simulation.get("lead_time_stats", {})
            weibull_leadtime_mean = lead_time_stats.get('mean', 0)
            weibull_leadtime_p95 = int(lead_time_stats.get('p95', 0))
            weibull_leadtime_p99 = int(lead_time_stats.get('p99', 0))
            
            # For backward compatibility, also get 10-year percentiles
            weibull_optimal_95 = weibull_simulation.get("optimal_stock_95", 0)
            weibull_optimal_99 = weibull_simulation.get("optimal_stock_99", 0)
            annual_failure_rate = weibull_simulation.get("annual_failure_rate", 0)
            
            # Calculate MTBF-based recommendations
            # Lead time demand using MTBF
            lead_time_demand_mtbf = (lead_time_days / 365) * (365 / mtbf) if mtbf > 0 else 0
            # Safety stock for desired service level
            z_score = norm.ppf(service_level)
            safety_stock_mtbf = z_score * np.sqrt(lead_time_demand_mtbf) if lead_time_demand_mtbf > 0 else 0
            # Total MTBF-based stock
            mtbf_optimal = int(np.ceil(lead_time_demand_mtbf + safety_stock_mtbf))
            # Calculate annual demand using MTBF
            annual_demand_mtbf = 365 / mtbf if mtbf > 0 else 0
            
            # Compare approaches
            comparison = {
                "weibull_95_service_10yr": weibull_optimal_95,
                "weibull_99_service_10yr": weibull_optimal_99,
                "weibull_leadtime_mean": weibull_leadtime_mean,
                "weibull_leadtime_p95": weibull_leadtime_p95,
                "weibull_leadtime_p99": weibull_leadtime_p99,
                "mtbf_based": mtbf_optimal,
                "annual_failure_rate_weibull": annual_failure_rate,
                "annual_demand_mtbf": annual_demand_mtbf,
                "recommended_approach": "Weibull" if weibull_leadtime_p95 > 0 else "MTBF",
                "recommended_stock": max(weibull_leadtime_p95, mtbf_optimal)
            }
            
            # Add recommendations based on failure pattern
            beta = weibull_simulation.get("weibull_params", {}).get("beta", 1.0)
            if beta < 1.0:
                comparison["failure_pattern"] = "Infant Mortality"
                comparison["recommendation"] = "High initial stock recommended due to infant mortality pattern"
            elif beta < 1.5:
                comparison["failure_pattern"] = "Random"
                comparison["recommendation"] = "Standard stocking levels appropriate for random failures"
            else:
                comparison["failure_pattern"] = "Wear Out"
                comparison["recommendation"] = "Consider increasing stock as equipment ages"
            
            # Add lead time stats for UI
            comparison["lead_time_stats"] = lead_time_stats
            
            return {
                "status": "Stocking levels calculated",
                "comparison": comparison,
                "lead_time_days": lead_time_days,
                "service_level": service_level
            }
            
        except Exception as e:
            logging.error(f"Error calculating optimal stocking levels: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def calculate_stockout_risk(self, current_stock: int, demand_forecast: Dict[str, Any], 
                              lead_time_days: int = 30) -> Dict[str, Any]:
        """Calculate stockout risk based on current stock and demand forecast"""
        try:
            if current_stock < 0:
                return {"status": "Invalid current stock"}
            
            # Get demand statistics
            mean_demand = demand_forecast.get("mean_demand", 0)
            std_demand = demand_forecast.get("std_demand", 0)
            percentiles = demand_forecast.get("percentiles", {})
            
            # Calculate lead time demand
            lead_time_demand_rate = mean_demand * (lead_time_days / 365)
            lead_time_demand_std = std_demand * np.sqrt(lead_time_days / 365)
            
            # Calculate stockout probability using normal approximation
            if lead_time_demand_std > 0:
                z_score = (current_stock - lead_time_demand_rate) / lead_time_demand_std
                stockout_probability = 1 - norm.cdf(z_score)
            else:
                stockout_probability = 1.0 if current_stock < lead_time_demand_rate else 0.0
            
            # Calculate service level
            service_level = 1 - stockout_probability
            
            # Determine risk level
            if stockout_probability < 0.05:
                risk_level = "Low"
            elif stockout_probability < 0.20:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            # Calculate recommended safety stock
            target_service_level = 0.95  # 95% service level
            safety_stock = norm.ppf(target_service_level) * lead_time_demand_std
            
            return {
                "status": "Analysis complete",
                "current_stock": current_stock,
                "lead_time_demand_rate": lead_time_demand_rate,
                "lead_time_demand_std": lead_time_demand_std,
                "stockout_probability": stockout_probability,
                "service_level": service_level,
                "risk_level": risk_level,
                "recommended_safety_stock": safety_stock,
                "recommended_reorder_point": lead_time_demand_rate + safety_stock
            }
            
        except Exception as e:
            logging.error(f"Error calculating stockout risk: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def calculate_economic_order_quantity(self, annual_demand: float, order_cost: float, 
                                        holding_cost_rate: float, unit_cost: float) -> Dict[str, Any]:
        """Calculate Economic Order Quantity (EOQ)"""
        try:
            if annual_demand <= 0 or order_cost <= 0 or holding_cost_rate <= 0 or unit_cost <= 0:
                return {"status": "Invalid input parameters"}
            
            # Calculate EOQ
            holding_cost_per_unit = unit_cost * holding_cost_rate
            eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
            
            # Calculate related metrics
            annual_orders = annual_demand / eoq
            order_cycle_days = 365 / annual_orders
            total_annual_cost = (annual_demand * unit_cost) + \
                              (annual_orders * order_cost) + \
                              (eoq / 2 * holding_cost_per_unit)
            
            return {
                "status": "Calculation complete",
                "eoq": eoq,
                "annual_orders": annual_orders,
                "order_cycle_days": order_cycle_days,
                "total_annual_cost": total_annual_cost,
                "holding_cost_per_unit": holding_cost_per_unit
            }
            
        except Exception as e:
            logging.error(f"Error calculating EOQ: {e}")
            return {"status": f"Error: {str(e)}"}
    
    def create_spares_analysis_plot(self, demand_analysis: Dict[str, Any], 
                                  simulation_results: Dict[str, Any],
                                  frame: ttk.Frame, title: str = "Spares Analysis") -> Any:
        """Create spares analysis visualization"""
        try:
            if not demand_analysis or demand_analysis.get("status") != "Analysis complete":
                # Create empty plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No spares analysis data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                return fig
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Demand by Failure Mode
            spares_demand = demand_analysis.get("spares_demand", {})
            if spares_demand:
                failure_codes = list(spares_demand.keys())[:10]  # Top 10
                demand_rates = [spares_demand[code]['demand_rate_per_year'] for code in failure_codes]
                
                bars = ax1.bar(range(len(failure_codes)), demand_rates, color='skyblue')
                ax1.set_title('Demand Rate by Failure Mode (Top 10)')
                ax1.set_xlabel('Failure Code')
                ax1.set_ylabel('Demand Rate (per year)')
                ax1.set_xticks(range(len(failure_codes)))
                ax1.set_xticklabels(failure_codes, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, demand_rates):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.1f}', ha='center', va='bottom')
            else:
                ax1.text(0.5, 0.5, "No demand data", ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Demand by Failure Mode')
            
            # Plot 2: Cost Analysis
            if spares_demand:
                costs = [spares_demand[code]['total_cost'] for code in failure_codes]
                avg_costs = [spares_demand[code]['avg_cost_per_demand'] for code in failure_codes]
                
                x = np.arange(len(failure_codes))
                width = 0.35
                
                ax2.bar(x - width/2, costs, width, label='Total Cost', color='lightcoral')
                ax2.bar(x + width/2, avg_costs, width, label='Avg Cost per Demand', color='lightgreen')
                
                ax2.set_title('Cost Analysis')
                ax2.set_xlabel('Failure Code')
                ax2.set_ylabel('Cost ($)')
                ax2.set_xticks(x)
                ax2.set_xticklabels(failure_codes, rotation=45, ha='right')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "No cost data", ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Cost Analysis')
            
            # Plot 3: Demand Simulation (years on x-axis)
            if simulation_results and simulation_results.get("status") == "Simulation complete":
                sim_data = simulation_results.get("simulation_results", [])
                if sim_data:
                    # Plot sample of simulation runs
                    days = np.arange(simulation_results.get("simulation_days", 365))
                    years = days / 365
                    for i in range(min(10, len(sim_data))):  # Plot first 10 runs
                        ax3.plot(years, sim_data[i], alpha=0.3, color='blue')
                    
                    # Plot mean and percentiles
                    sim_array = np.array(sim_data)
                    mean_line = np.mean(sim_array, axis=0)
                    p95_line = np.percentile(sim_array, 95, axis=0)
                    
                    ax3.plot(years, mean_line, 'r-', linewidth=2, label='Mean')
                    ax3.plot(years, p95_line, 'g--', linewidth=2, label='95th Percentile')
                    
                    ax3.set_title('Demand Simulation (Years)')
                    ax3.set_xlabel('Years')
                    ax3.set_ylabel('Cumulative Demand')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, "No simulation data", ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Demand Simulation')
            else:
                ax3.text(0.5, 0.5, "No simulation data", ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Demand Simulation')
            
            # Plot 4: Sensitivity Chart for Reliability Levels
            if simulation_results and simulation_results.get("status") == "Simulation complete":
                percentiles = simulation_results.get("percentiles", {})
                if percentiles:
                    # Create stockout risk visualization
                    reliability_levels = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
                    spares_needed = []
                    for r in reliability_levels:
                        p = int(np.percentile([sim[-1] for sim in simulation_results.get("simulation_results", [])], 100*r))
                        spares_needed.append(p)
                    ax4.plot([int(r*100) for r in reliability_levels], spares_needed, marker='o', color='purple')
                    ax4.set_title('Sensitivity: Spares vs. Reliability')
                    ax4.set_xlabel('Reliability Level (%)')
                    ax4.set_ylabel('Required Spares (10 yrs)')
                    ax4.set_xticks([int(r*100) for r in reliability_levels])
                    ax4.invert_xaxis()
                    for x, y in zip([int(r*100) for r in reliability_levels], spares_needed):
                        ax4.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, "No risk data", ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Sensitivity: Spares vs. Reliability')
            else:
                ax4.text(0.5, 0.5, "No sensitivity data", ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Sensitivity: Spares vs. Reliability')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            fig.suptitle(title, fontsize=16)
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            return fig
            
        except Exception as e:
            logging.error(f"Error creating spares analysis plot: {e}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error creating spares analysis plot: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(title)
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return fig
    
    def get_spares_recommendations(self, demand_analysis: Dict[str, Any], 
                                 simulation_results: Dict[str, Any],
                                 current_stock: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Generate spares management recommendations"""
        try:
            if not demand_analysis or demand_analysis.get("status") != "Analysis complete":
                return {"status": "No data available for recommendations"}
            
            spares_demand = demand_analysis.get("spares_demand", {})
            recommendations = []
            
            # Analyze each failure mode
            for failure_code, data in spares_demand.items():
                demand_rate = data['demand_rate_per_year']
                total_cost = data['total_cost']
                avg_cost = data['avg_cost_per_demand']
                
                # Get current stock for this item
                current_stock_level = current_stock.get(failure_code, 0) if current_stock else 0
                
                # Generate recommendations based on demand and cost
                if demand_rate > 5:
                    recommendations.append({
                        'failure_code': failure_code,
                        'priority': 'High',
                        'recommendation': 'High demand item - maintain adequate safety stock',
                        'demand_rate': demand_rate,
                        'current_stock': current_stock_level,
                        'suggested_stock': max(3, int(demand_rate * 0.5))  # 6 months of demand
                    })
                elif demand_rate > 1:
                    recommendations.append({
                        'failure_code': failure_code,
                        'priority': 'Medium',
                        'recommendation': 'Moderate demand - review stock levels quarterly',
                        'demand_rate': demand_rate,
                        'current_stock': current_stock_level,
                        'suggested_stock': max(1, int(demand_rate * 0.3))  # 4 months of demand
                    })
                else:
                    recommendations.append({
                        'failure_code': failure_code,
                        'priority': 'Low',
                        'recommendation': 'Low demand item - consider just-in-time ordering',
                        'demand_rate': demand_rate,
                        'current_stock': current_stock_level,
                        'suggested_stock': 0
                    })
                
                # Add cost-based recommendations
                if avg_cost > 10000:
                    recommendations[-1]['recommendation'] += ' - High value item, secure storage recommended'
                elif avg_cost < 100:
                    recommendations[-1]['recommendation'] += ' - Low value item, bulk ordering may be cost-effective'
            
            # Sort by priority
            priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
            
            return {
                "status": "Recommendations generated",
                "recommendations": recommendations,
                "total_items": len(recommendations),
                "high_priority_items": len([r for r in recommendations if r['priority'] == 'High']),
                "medium_priority_items": len([r for r in recommendations if r['priority'] == 'Medium']),
                "low_priority_items": len([r for r in recommendations if r['priority'] == 'Low'])
            }
            
        except Exception as e:
            logging.error(f"Error generating spares recommendations: {e}")
            return {"status": f"Error: {str(e)}"} 