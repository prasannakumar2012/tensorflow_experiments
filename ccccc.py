data = spark.read.format("com.databricks.spark.csv").option("header", "true").\
    option("inferSchema", "true").option("delimiter", '|').load("/Users/prasanna/Downloads/ds_tech_review_dataset.txt")