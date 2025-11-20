from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, monotonically_increasing_id, when
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# parquet files with date column dropped by dropdate.py
BASE = "/gpfs/projects/AMS598/class2025/Shaikh_Tasfia/ams598_netflixrecsys/data/processed"


def main():
    # spark session setup
    spark = (
        SparkSession.builder
        .appName("NetflixALS_CF")
        .master("local[8]")   # use 8 cores
        .config("spark.driver.memory", "40g")
        .config("spark.executor.memory", "40g")
        .config("spark.sql.shuffle.partitions", "400")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # load data
    ratings = (
        spark.read.parquet(f"{BASE}/ratings_train_no_probe.parquet")
             .select("user_id", "movie_id", "rating")
    )

    probe = (
        spark.read.parquet(f"{BASE}/probe_ratings.parquet")
             .select("user_id", "movie_id", "rating")
    )

    qual = (
        spark.read.parquet(f"{BASE}/qualifying_to_predict.parquet")
             .select("user_id", "movie_id")
    )

    # Spread ratings across more partitions to reduce per-task memory pressure
    ratings = ratings.repartition(400, "user_id")

    # sanity checks (will show up in netflix_als.out)
    print("ratings schema:")
    ratings.printSchema()
    print("probe schema:")
    probe.printSchema()
    print("qual schema:")
    qual.printSchema()

    # als model
    als = ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        rank=30,          # was 50; 30 is still reasonable but lighter in memory
        regParam=0.1,
        maxIter=10,
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop",  # drop NaN predictions during evaluation
    )

    print("Fitting ALS model ...")
    model = als.fit(ratings)
    print("ALS training finished.")

    # evaluate RMSE
    print("Evaluating on probe set ...")
    pred_probe = model.transform(probe)  # adds "prediction" column

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )
    rmse = evaluator.evaluate(pred_probe)
    print(f"\n*** Probe RMSE (ALS) = {rmse:.4f} ***\n")

    # Predict on qualifying_to_predict
    qual_with_id = qual.withColumn("row_id", monotonically_increasing_id())

    pred_qual = model.transform(qual_with_id)

    # Handle cold-start (users/movies unseen in training → prediction = null)
    global_mean = ratings.agg(avg("rating").alias("mean_rating")).collect()[0]["mean_rating"]
    print(f"Global mean rating (fallback) = {global_mean:.4f}")

    pred_qual = pred_qual.withColumn(
        "pred_rating",
        when(col("prediction").isNull(), global_mean).otherwise(col("prediction")),
    )

    # Order back by row_id to match qualifying_to_predict order
    final_pred = (
        pred_qual
        .orderBy("row_id")
        .select("movie_id", "user_id", "pred_rating")
    )

    # save predictions
    out_dir = (
        "/gpfs/projects/AMS598/class2025/"
        "Shaikh_Tasfia/ams598_netflixrecsys/collaborative_filtering/als_qual_predictions"
    )

    print(f"Saving qualifying predictions to {out_dir} (CSV, coalesced to 1 file) ...")
    (
        final_pred
        .coalesce(1)  
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(out_dir)
    )

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
