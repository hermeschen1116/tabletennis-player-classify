from typing import Any

import numpy
import polars
from datasets import Dataset, load_dataset
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

dataset = load_dataset(
	"csv", data_dir="data", data_files={"train": "train.csv", "test": "test.csv"}, keep_in_memory=True, num_proc=8
)

dataset_split = dataset["train"].train_test_split(test_size=0.1, keep_in_memory=True)
dataset["train"], dataset["validation"] = dataset_split["train"], dataset_split["test"]

index_columns: list = ["data_ID", "player_ID"]
label_columns: list = ["gender", "hold racket handed", "play years", "level"]
data_columns: list = list(set(dataset["train"].column_names).difference(set(index_columns + label_columns)))

scaler = MinMaxScaler()
scaler.fit(dataset["train"].to_polars()[data_columns].to_numpy())

train_dataset: polars.DataFrame = dataset["train"].to_polars()
train_data: numpy.ndarray = scaler.transform(train_dataset[data_columns].to_numpy())
train_label: dict[str, numpy.ndarray] = {label: train_dataset[label].to_numpy() for label in label_columns}

hyperparameter_grid: dict[str, list[Any]] = {
	"criterion": ["gini", "entropy", "log_loss"],  # 分裂判斷標準
	"splitter": ["best", "random"],  # 選擇分裂的策略
	"max_depth": [None, 10, 20, 30, 40, 50],  # 樹的最大深度
	"min_samples_split": [2, 5, 10],  # 分裂所需的最小樣本數
	"min_samples_leaf": [1, 2, 4],  # 葉節點的最小樣本數
	"min_weight_fraction_leaf": [0.0, 0.1, 0.2],  # 葉節點的最小樣本權重比
	"max_features": [None, "sqrt", "log2"],  # 每次分裂時的最大特徵數
	"max_leaf_nodes": [None, 10, 20, 30],  # 最大葉節點數
	"min_impurity_decrease": [0.0, 0.1, 0.2],  # 節點分裂所需的最小不純度減少
	"class_weight": [None, "balanced"],  # 類別權重
	"ccp_alpha": [0.0, 0.1, 0.2],  # 剪枝的複雜度參數
}

best_classifiers: dict = {}
for label in label_columns:
	print(f"Tune classifier for {label}")
	classifier = DecisionTreeClassifier(random_state=37)

	scoring: str = "roc_auc" if label in ["gender", "hold racket handed"] else "roc_auc_ovr"
	tuner = RandomizedSearchCV(
		classifier, hyperparameter_grid, n_iter=100, cv=5, scoring=scoring, n_jobs=-1, verbose=10
	)

	tuner.fit(train_data[label], train_label[label])

	best_classifiers[label] = tuner.best_estimator_

test_data: polars.DataFrame = scaler.transform(dataset["test"].select_columns(data_columns).to_polars().to_numpy())

predictions: dict = {"data_ID": dataset["test"]["data_ID"]}
for label in label_columns:
	predictions[label] = best_classifiers[label].predict(test_data).tolist()

test_result = Dataset.from_dict(predictions)

test_result = test_result.map(
	lambda samples: {
		"play years_0": [1 if sample == 0 else 0 for sample in samples],
		"play years_1": [1 if sample == 1 else 0 for sample in samples],
		"play years_2": [1 if sample == 2 else 0 for sample in samples],
	},
	input_columns=["play years"],
	remove_columns=["play years"],
	batched=True,
	num_proc=8,
)

test_result = test_result.map(
	lambda samples: {
		"level_0": [1 if sample == 0 else 0 for sample in samples],
		"level_1": [1 if sample == 1 else 0 for sample in samples],
		"level_2": [1 if sample == 2 else 0 for sample in samples],
	},
	input_columns=["level"],
	remove_columns=["level"],
	batched=True,
	num_proc=8,
)

test_result.to_csv("../dist/result_dt.csv", num_proc=8)
