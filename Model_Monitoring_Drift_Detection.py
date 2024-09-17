from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable

report = Report(metrics=[DataDriftTable()])
column_mapping = ColumnMapping(target='target_column_name', prediction='prediction_column_name')

report.run(reference_data=X_test, current_data=X_new, column_mapping=column_mapping)

report.save_html('data_drift_report.html')

#Model Monitoring and Drift Detection
