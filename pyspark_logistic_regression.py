from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
import pyspark  # for type hint
from pyspark.sql import SparkSession

# Pyspark leverages Resilient Distributed Dataset (RDD) for datasets that need to be partitioned on a cluster
# They have their own dataframe object which is a little different from pandas dataframes
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.html

# https://sparkbyexamples.com/pyspark/pyspark-what-is-sparksession/
spark = SparkSession.builder.master("local[1]") \
                    .appName('jdk_spark_test') \
                    .getOrCreate()

columns = ["user_id", "month_interaction_count", "week_interaction_count", "day_interaction_count", "cancelled_within_week"]
rows = [("010b4076", 31, 11, 2, 0),
        ("31c73683", 29, 11, 2, 0),
        ("8173164f", 30, 11, 2, 0),
        ("f77ad2d3", 28, 6, 2, 0),
        ("25050522", 29, 8, 4, 0),
        ("bfb27c75", 33, 8, 2, 0),
        ("09663ea6", 33, 7, 0, 1),
        ("ca7aacf2", 20, 1, 0, 1),
        ("63f84e80", 33, 8, 3, 0),
        ("cbb81ed7", 24, 9, 3, 0)]

pyspark_df = spark.createDataFrame(rows).toDF(*columns)


def predict_cancellations(user_interaction_df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    # DataFrame[user_id: string, month_interaction_count: bigint, week_interaction_count: bigint,
    # day_interaction_count: bigint, cancelled_within_week: int]
    print("row count", user_interaction_df.count())
    user_interaction_df.show()
    #
    # https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html
    # combine pyspark dataframe inputs into one column/vector called "features"; this is what the model will look for by default
    v_assembler = VectorAssembler(inputCols=["month_interaction_count", "week_interaction_count", "day_interaction_count"],
                                  outputCol="features")

    input_df = v_assembler.transform(user_interaction_df)
    # we also need to specify a "label" column
    input_df = input_df.withColumn("label", input_df["cancelled_within_week"])

    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html
    # 10 iterations/epochs?, decision threshold, L1 regularization (penalty: sum(weights)*0.1) (aka Lasso Regression)
    # elasticNetParam=0 is L2 regularization. And a number between is a blend of L1 and L2. (fractional exponent between 1 and 2?)
    # L1 regularization can push feature coefficients/weights to 0, creating a method for feature selection.
    # Regularization for logistic regression seems more helpful for feature rich data
    # pyspark regression threshold note/question/confusion:
    # it seems that
    # if probability > 1 - threshold => predict 0
    # not
    # if probability > threshold => predict 1
    # which we'd expect
    model = LogisticRegression(maxIter=10, threshold=0.6, elasticNetParam=1, regParam=0.1)
    trained_model = model.fit(input_df)

    # this is sort of like getting the training accuracy; it will append some prediction features to the data frame
    output_df = trained_model.transform(input_df)
    # we'll just take user_id, log(odds) = z, probability (1/(1+exp(-z)), final prediction 0/1 (and not the features)
    output_df = output_df.select(["user_id", "rawPrediction", "probability", "prediction"])
    print("output predictions")
    output_df.show()

    return output_df


predict_cancellations(pyspark_df)

# input:
# +--------+-----------------------+----------------------+---------------------+---------------------+
# | user_id|month_interaction_count|week_interaction_count|day_interaction_count|cancelled_within_week|
# +--------+-----------------------+----------------------+---------------------+---------------------+
# |010b4076|                     31|                    11|                    2|                    0|
# |31c73683|                     29|                    11|                    2|                    0|
# |8173164f|                     30|                    11|                    2|                    0|
# |f77ad2d3|                     28|                     6|                    2|                    0|
# |25050522|                     29|                     8|                    4|                    0|
# |bfb27c75|                     33|                     8|                    2|                    0|
# |09663ea6|                     33|                     7|                    0|                    1|
# |ca7aacf2|                     20|                     1|                    0|                    1|
# |63f84e80|                     33|                     8|                    3|                    0|
# |cbb81ed7|                     24|                     9|                    3|                    0|
# +--------+-----------------------+----------------------+---------------------+---------------------+