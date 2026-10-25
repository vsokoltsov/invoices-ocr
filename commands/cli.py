from app.extraction.bert.pipeline import BertTrainingPipeline


def main():
    train_pipeline = BertTrainingPipeline(
        data_path='data/raw/',
        artifacts_path='data/interim/'
    )
    train_pipeline.run()


if __name__ == '__main__':
    main()

