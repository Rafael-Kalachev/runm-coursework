import os
database_file = "diabetes-dataset.xlsx"
train_test_split_percent=0.3
run_name="test_003"






out_base_dir=os.path.join(".", 'out')
out_dir=os.path.join(out_base_dir, run_name)

run_log_file_path=os.path.join(out_dir, "run_log.log")
