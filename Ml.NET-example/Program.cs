using Microsoft.ML;
using Microsoft.ML.Data;

namespace Ml.NET_example
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Load data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TitanicData>("data.csv", hasHeader: true, separatorChar: ',');

            // Data processing configuration with a data transformation pipeline
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SexEncoded", inputColumnName: nameof(TitanicData.Sex))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(TitanicData.PClass), "SexEncoded", nameof(TitanicData.Age)))
                .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(TitanicData.Survived)));

            // Set the training algorithm
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model on the dataset
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // Evaluate the model on the training set
            var predictions = trainedModel.Transform(trainingDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            // Use the trained model for a single prediction
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TitanicData, TitanicPrediction>(trainedModel);

            var prediction = predictionFunction.Predict(new TitanicData()
            {
                PClass = 2,
                Sex = "female",
                Age = 30
            });

            Console.WriteLine($"Predicted chance to survive: {prediction.Survived:0.##}");
        }

        public class TitanicData
        {
            [LoadColumn(0)]
            public float Survived { get; set; }

            [LoadColumn(1)]
            public float PClass { get; set; }

            [LoadColumn(2)]
            public string Sex { get; set; }

            [LoadColumn(3)]
            public float Age { get; set; }
        }

        public class TitanicPrediction
        {
            [ColumnName("Score")]
            public float Survived { get; set; }
        }
    }
}