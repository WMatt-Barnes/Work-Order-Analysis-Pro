"""
FMEA Export Module
Provides functionality to export failure mode data for FMEA analysis
with calculated parameters and risk priority numbers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import os

class FMEAExport:
    """FMEA export functionality for failure mode analysis"""
    
    def __init__(self):
        self.failure_modes = {}
        self.equipment_data = {}
        self.weibull_analyses = {}
        
    def aggregate_failure_modes(self, df: pd.DataFrame, weibull_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Aggregate failure modes with frequency and cost data"""
        try:
            if df.empty:
                return {}
            
            # Group by failure code and description
            failure_groups = df.groupby(['Failure Code', 'Failure Description']).agg({
                'Work Order': 'count',
                'Work Order Cost': 'sum',
                'Equipment #': lambda x: list(x.unique()),
                'Reported Date': ['min', 'max'],
                'Asset': lambda x: list(x.unique())
            }).reset_index()
            
            # Flatten column names
            failure_groups.columns = ['Failure Code', 'Failure Description', 'Frequency', 
                                    'Total Cost', 'Equipment List', 'First Failure', 'Last Failure', 'Asset List']
            
            # Calculate additional metrics
            failure_modes = {}
            for _, row in failure_groups.iterrows():
                failure_code = row['Failure Code']
                
                # Calculate time span
                first_failure = pd.to_datetime(row['First Failure'])
                last_failure = pd.to_datetime(row['Last Failure'])
                time_span_days = (last_failure - first_failure).days
                
                # Calculate failure rate per year
                if time_span_days > 0:
                    failure_rate_per_year = (row['Frequency'] / time_span_days) * 365
                else:
                    failure_rate_per_year = 0
                
                # Calculate average cost per failure
                avg_cost_per_failure = row['Total Cost'] / row['Frequency'] if row['Frequency'] > 0 else 0
                
                # Get Weibull parameters if available
                weibull_params = None
                if weibull_results and failure_code in weibull_results:
                    weibull_params = weibull_results[failure_code]
                
                failure_modes[failure_code] = {
                    'failure_code': failure_code,
                    'failure_description': row['Failure Description'],
                    'frequency': int(row['Frequency']),
                    'total_cost': float(row['Total Cost']),
                    'avg_cost_per_failure': float(avg_cost_per_failure),
                    'failure_rate_per_year': float(failure_rate_per_year),
                    'time_span_days': int(time_span_days),
                    'equipment_count': len(row['Equipment List']),
                    'equipment_list': row['Equipment List'],
                    'asset_list': row['Asset List'],
                    'first_failure': row['First Failure'],
                    'last_failure': row['Last Failure'],
                    'weibull_beta': weibull_params.get('beta', 1.0) if weibull_params else 1.0,
                    'weibull_eta': weibull_params.get('eta', 365.0) if weibull_params else 365.0,
                    'reliability_30d': weibull_params.get('reliability_30d', 0.0) if weibull_params else 0.0,
                    'reliability_90d': weibull_params.get('reliability_90d', 0.0) if weibull_params else 0.0,
                    'reliability_365d': weibull_params.get('reliability_365d', 0.0) if weibull_params else 0.0
                }
            
            logging.info(f"Aggregated {len(failure_modes)} failure modes")
            return failure_modes
            
        except Exception as e:
            logging.error(f"Error aggregating failure modes: {e}")
            return {}
    
    def calculate_rpn(self, failure_modes: Dict[str, Any], 
                     severity_weights: Dict[str, int] = None,
                     occurrence_weights: Dict[str, int] = None,
                     detection_weights: Dict[str, int] = None) -> Dict[str, Any]:
        """Calculate Risk Priority Number (RPN) for each failure mode"""
        try:
            if not failure_modes:
                return {}
            
            # Default severity weights based on failure rate and cost
            if severity_weights is None:
                severity_weights = {
                    'high_cost': 10,    # Cost > $10,000
                    'medium_cost': 7,   # Cost $1,000-$10,000
                    'low_cost': 4,      # Cost < $1,000
                    'high_frequency': 10,  # > 5 failures/year
                    'medium_frequency': 7, # 1-5 failures/year
                    'low_frequency': 4     # < 1 failure/year
                }
            
            # Default occurrence weights based on failure rate
            if occurrence_weights is None:
                occurrence_weights = {
                    'very_high': 10,  # > 10 failures/year
                    'high': 8,        # 5-10 failures/year
                    'medium': 6,      # 2-5 failures/year
                    'low': 4,         # 0.5-2 failures/year
                    'very_low': 2     # < 0.5 failures/year
                }
            
            # Default detection weights (simplified)
            if detection_weights is None:
                detection_weights = {
                    'very_low': 10,   # No detection method
                    'low': 8,         # Poor detection
                    'medium': 6,      # Moderate detection
                    'high': 4,        # Good detection
                    'very_high': 2    # Excellent detection
                }
            
            rpn_results = {}
            
            for failure_code, failure_data in failure_modes.items():
                # Calculate severity based on cost and frequency
                total_cost = failure_data['total_cost']
                failure_rate = failure_data['failure_rate_per_year']
                
                # Cost-based severity
                if total_cost > 10000:
                    cost_severity = severity_weights['high_cost']
                elif total_cost > 1000:
                    cost_severity = severity_weights['medium_cost']
                else:
                    cost_severity = severity_weights['low_cost']
                
                # Frequency-based severity
                if failure_rate > 5:
                    freq_severity = severity_weights['high_frequency']
                elif failure_rate > 1:
                    freq_severity = severity_weights['medium_frequency']
                else:
                    freq_severity = severity_weights['low_frequency']
                
                # Overall severity (average of cost and frequency)
                severity = max(cost_severity, freq_severity)
                
                # Calculate occurrence based on failure rate
                if failure_rate > 10:
                    occurrence = occurrence_weights['very_high']
                elif failure_rate > 5:
                    occurrence = occurrence_weights['high']
                elif failure_rate > 2:
                    occurrence = occurrence_weights['medium']
                elif failure_rate > 0.5:
                    occurrence = occurrence_weights['low']
                else:
                    occurrence = occurrence_weights['very_low']
                
                # Detection (simplified - could be enhanced with more data)
                # Assume moderate detection for now
                detection = detection_weights['medium']
                
                # Calculate RPN
                rpn = severity * occurrence * detection
                
                rpn_results[failure_code] = {
                    **failure_data,
                    'severity': severity,
                    'occurrence': occurrence,
                    'detection': detection,
                    'rpn': rpn,
                    'risk_level': self._get_risk_level(rpn)
                }
            
            # Sort by RPN (highest first)
            sorted_rpn = dict(sorted(rpn_results.items(), key=lambda x: x[1]['rpn'], reverse=True))
            
            logging.info(f"Calculated RPN for {len(sorted_rpn)} failure modes")
            return sorted_rpn
            
        except Exception as e:
            logging.error(f"Error calculating RPN: {e}")
            return {}
    
    def _get_risk_level(self, rpn: int) -> str:
        """Get risk level based on RPN value"""
        if rpn >= 200:
            return "Critical"
        elif rpn >= 100:
            return "High"
        elif rpn >= 50:
            return "Medium"
        else:
            return "Low"
    
    def create_fmea_dataframe(self, rpn_results: Dict[str, Any]) -> pd.DataFrame:
        """Create DataFrame for FMEA export"""
        try:
            if not rpn_results:
                return pd.DataFrame()
            
            fmea_data = []
            for failure_code, data in rpn_results.items():
                fmea_data.append({
                    'Failure Code': data['failure_code'],
                    'Failure Description': data['failure_description'],
                    'Frequency': data['frequency'],
                    'Total Cost ($)': f"${data['total_cost']:,.2f}",
                    'Avg Cost per Failure ($)': f"${data['avg_cost_per_failure']:,.2f}",
                    'Failure Rate (per year)': f"{data['failure_rate_per_year']:.2f}",
                    'Equipment Count': data['equipment_count'],
                    'Weibull β (Shape)': f"{data['weibull_beta']:.3f}",
                    'Weibull η (Scale)': f"{data['weibull_eta']:.1f}",
                    'Reliability 30d': f"{data['reliability_30d']:.3f}",
                    'Reliability 90d': f"{data['reliability_90d']:.3f}",
                    'Reliability 365d': f"{data['reliability_365d']:.3f}",
                    'Severity': data['severity'],
                    'Occurrence': data['occurrence'],
                    'Detection': data['detection'],
                    'RPN': data['rpn'],
                    'Risk Level': data['risk_level'],
                    'Equipment List': ', '.join(data['equipment_list']),
                    'Asset List': ', '.join(data['asset_list']),
                    'First Failure': data['first_failure'],
                    'Last Failure': data['last_failure']
                })
            
            df = pd.DataFrame(fmea_data)
            logging.info(f"Created FMEA DataFrame with {len(df)} rows")
            return df
            
        except Exception as e:
            logging.error(f"Error creating FMEA DataFrame: {e}")
            return pd.DataFrame()
    
    def export_fmea_excel(self, rpn_results: Dict[str, Any], output_path: str, 
                         include_summary: bool = True) -> bool:
        """Export FMEA data to Excel with multiple sheets"""
        try:
            if not rpn_results:
                logging.warning("No FMEA data to export")
                return False
            
            # Create FMEA DataFrame
            fmea_df = self.create_fmea_dataframe(rpn_results)
            
            # Create summary statistics
            summary_data = []
            if include_summary:
                total_failures = sum(data['frequency'] for data in rpn_results.values())
                total_cost = sum(data['total_cost'] for data in rpn_results.values())
                avg_rpn = np.mean([data['rpn'] for data in rpn_results.values()])
                
                # Risk level distribution
                risk_levels = {}
                for data in rpn_results.values():
                    level = data['risk_level']
                    risk_levels[level] = risk_levels.get(level, 0) + 1
                
                summary_data = [
                    {'Metric': 'Total Failure Modes', 'Value': len(rpn_results)},
                    {'Metric': 'Total Failures', 'Value': total_failures},
                    {'Metric': 'Total Cost ($)', 'Value': f"${total_cost:,.2f}"},
                    {'Metric': 'Average RPN', 'Value': f"{avg_rpn:.1f}"},
                    {'Metric': 'Critical Risk Modes', 'Value': risk_levels.get('Critical', 0)},
                    {'Metric': 'High Risk Modes', 'Value': risk_levels.get('High', 0)},
                    {'Metric': 'Medium Risk Modes', 'Value': risk_levels.get('Medium', 0)},
                    {'Metric': 'Low Risk Modes', 'Value': risk_levels.get('Low', 0)}
                ]
            
            # Export to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main FMEA data
                fmea_df.to_excel(writer, sheet_name='FMEA Data', index=False)
                
                # Summary sheet
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Risk priority sheet (top 20 by RPN)
                top_risks = dict(list(rpn_results.items())[:20])
                top_risks_df = self.create_fmea_dataframe(top_risks)
                top_risks_df.to_excel(writer, sheet_name='Top 20 Risks', index=False)
                
                # Equipment analysis sheet
                equipment_analysis = self._create_equipment_analysis(rpn_results)
                if not equipment_analysis.empty:
                    equipment_analysis.to_excel(writer, sheet_name='Equipment Analysis', index=False)
            
            logging.info(f"FMEA data exported to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error exporting FMEA data: {e}")
            return False
    
    def _create_equipment_analysis(self, rpn_results: Dict[str, Any]) -> pd.DataFrame:
        """Create equipment-level analysis for FMEA export"""
        try:
            equipment_data = {}
            
            for failure_code, data in rpn_results.items():
                for equipment in data['equipment_list']:
                    if equipment not in equipment_data:
                        equipment_data[equipment] = {
                            'Equipment': equipment,
                            'Failure Modes': 0,
                            'Total Failures': 0,
                            'Total Cost': 0.0,
                            'Avg RPN': 0.0,
                            'Critical Risks': 0,
                            'High Risks': 0
                        }
                    
                    equipment_data[equipment]['Failure Modes'] += 1
                    equipment_data[equipment]['Total Failures'] += data['frequency']
                    equipment_data[equipment]['Total Cost'] += data['total_cost']
                    
                    if data['risk_level'] == 'Critical':
                        equipment_data[equipment]['Critical Risks'] += 1
                    elif data['risk_level'] == 'High':
                        equipment_data[equipment]['High Risks'] += 1
            
            # Calculate average RPN for each equipment
            for equipment in equipment_data:
                equipment_failures = [data for data in rpn_results.values() 
                                    if equipment in data['equipment_list']]
                if equipment_failures:
                    avg_rpn = np.mean([data['rpn'] for data in equipment_failures])
                    equipment_data[equipment]['Avg RPN'] = avg_rpn
            
            # Convert to DataFrame and sort by total cost
            df = pd.DataFrame(list(equipment_data.values()))
            if not df.empty:
                df = df.sort_values('Total Cost', ascending=False)
                df['Total Cost ($)'] = df['Total Cost'].apply(lambda x: f"${x:,.2f}")
                df['Avg RPN'] = df['Avg RPN'].apply(lambda x: f"{x:.1f}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error creating equipment analysis: {e}")
            return pd.DataFrame()
    
    def get_fmea_summary(self, rpn_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics for FMEA analysis"""
        try:
            if not rpn_results:
                return {"status": "No data available"}
            
            total_failures = sum(data['frequency'] for data in rpn_results.values())
            total_cost = sum(data['total_cost'] for data in rpn_results.values())
            avg_rpn = np.mean([data['rpn'] for data in rpn_results.values()])
            max_rpn = max([data['rpn'] for data in rpn_results.values()])
            
            # Risk level distribution
            risk_levels = {}
            for data in rpn_results.values():
                level = data['risk_level']
                risk_levels[level] = risk_levels.get(level, 0) + 1
            
            # Top 5 failure modes by RPN
            top_failures = sorted(rpn_results.items(), key=lambda x: x[1]['rpn'], reverse=True)[:5]
            top_failure_list = [
                {
                    'code': data['failure_code'],
                    'description': data['failure_description'],
                    'rpn': data['rpn'],
                    'risk_level': data['risk_level']
                }
                for _, data in top_failures
            ]
            
            return {
                "status": "Analysis complete",
                "total_failure_modes": len(rpn_results),
                "total_failures": total_failures,
                "total_cost": total_cost,
                "average_rpn": avg_rpn,
                "max_rpn": max_rpn,
                "risk_level_distribution": risk_levels,
                "top_failures": top_failure_list
            }
            
        except Exception as e:
            logging.error(f"Error getting FMEA summary: {e}")
            return {"status": f"Error: {str(e)}"} 