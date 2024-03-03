using Microsoft.ML;
using Microsoft.ML.Data;

namespace Ml.NET_example
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Загрузить данные
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<HousingData>("data.csv", hasHeader: true, separatorChar: ',');

            // Конфигурация обработки данных с использованием конвейера преобразований данных
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", nameof(HousingData.Floor), nameof(HousingData.NumberOfRooms), nameof(HousingData.InfrastructureScore))
                .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(HousingData.Price)));

            // Установить алгоритм обучения
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Обучить модель на наборе данных
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // Оценить модель на обучающем наборе
            var predictions = trainedModel.Transform(trainingDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            // Использовать обученную модель для одного прогноза
            var predictionFunction = mlContext.Model.CreatePredictionEngine<HousingData, HousingPrediction>(trainedModel);

            var prediction = predictionFunction.Predict(new HousingData()
            {
                Floor = 2,
                NumberOfRooms = 3,
                InfrastructureScore = 8
            });

            Console.WriteLine($"Predicted price: {prediction.Price:0.##}");
        }

        public class HousingData
        {
            [LoadColumn(0)]
            public float Floor { get; set; }

            [LoadColumn(1)]
            public float NumberOfRooms { get; set; }

            [LoadColumn(2)]
            public float InfrastructureScore { get; set; }

            [LoadColumn(3)]
            public float Price { get; set; }
        }

        public class HousingPrediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }
    }
}