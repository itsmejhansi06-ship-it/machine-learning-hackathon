import pandas as pd

runs = pd.read_csv("results/model_runs.csv").drop_duplicates("model", keep="last")  # latest run per model
table = runs[["model", "family", "config", "cv_macro_f1_mean", "cv_macro_f1_std",
              "train_top1_acc", "test_top1_acc", "test_macro_f1", "test_precision",
              "test_recall", "test_log_loss", "test_continuity",
              "test_train_time_s", "test_infer_ms_per_obs"]]
table.to_csv("results/model_comparison.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 20)
print(table.drop(columns="config").round(4).to_string(index=False))