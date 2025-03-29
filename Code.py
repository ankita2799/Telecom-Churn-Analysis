from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, floor, avg, count, sum, round
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize SparkSession
spark = SparkSession.builder.appName("TelcoChurn").getOrCreate()

# Phase 1: Data Acquisition and Understanding
# Load the data
df = spark.read.csv("telco_churn.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)

# Phase 2: Data Preparation
# Data Cleaning
df = df.withColumn("TotalCharges", when(col("TotalCharges") == " ", None).otherwise(col("TotalCharges").cast("double")))
df = df.dropna(subset=["TotalCharges"])

# Feature Engineering
df = df.withColumn("TenureGroup", floor(col("tenure") / 12))
df = df.withColumn("InternetAndPhone", when((col("InternetService").isNotNull()) & (col("PhoneService") == "Yes"), "Yes").otherwise("No"))

# Spark SQL
df.createOrReplaceTempView("telco_churn")

# Example SQL queries
spark.sql("SELECT gender, AVG(MonthlyCharges) FROM telco_churn GROUP BY gender").show()
spark.sql("select Contract, avg(case when Churn = 'Yes' then 1 else 0 end) as churn_rate from telco_churn group by Contract").show()

# Phase 3: Exploratory Data Analysis (EDA)

# Univariate Analysis
df.describe().show()

# Convert to Pandas for visualization
pandas_df = df.toPandas()

# Example visualizations (Matplotlib and Seaborn)
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=pandas_df)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(pandas_df['MonthlyCharges'], kde=True)
plt.show()

# Bivariate Analysis
plt.figure(figsize=(12, 6))
sns.countplot(x='Contract', hue='Churn', data=pandas_df)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=pandas_df)
plt.show()

# Phase 4: Customer Segmentation (Optional)
# Example rule based segmentation
df = df.withColumn("HighValue", when(col("MonthlyCharges") > 80, "Yes").otherwise("No"))

# Phase 5: Churn Analysis and Insights
# Insights from EDA are gathered from the printed charts and tables.

# Phase 6: Predictive Modeling
# Feature Preparation
categorical_cols = [item[0] for item in df.dtypes if item[1].startswith('string') and item[0] != 'customerID' and item[0] != 'Churn']
numerical_cols = [item[0] for item in df.dtypes if item[1].startswith('double') or item[1].startswith('int')]

stages = []

for categoricalCol in categorical_cols:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

assemblerInputs = [c + "classVec" for c in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
stages += [scaler]

label_stringIdx = StringIndexer(inputCol="Churn", outputCol="label")
stages += [label_stringIdx]

# Model Selection and Training
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label')
rf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='label')
gbt = GBTClassifier(featuresCol='scaledFeatures', labelCol='label')

stages += [lr]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
prediction = pipelineModel.transform(df)

# Model Evaluation
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
auc = evaluator.evaluate(prediction)
print("Logistic Regression AUC:", auc)

stages.pop() # remove last model from pipeline
stages.append(rf)
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
prediction = pipelineModel.transform(df)
auc = evaluator.evaluate(prediction)
print("Random Forest AUC:", auc)

stages.pop()
stages.append(gbt)
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
prediction = pipelineModel.transform(df)
auc = evaluator.evaluate(prediction)
print("GBT AUC:", auc)

# Stop SparkSession
spark.stop()
