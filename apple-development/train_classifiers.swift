//  train_classifiers.swift
//  Trains one CreateML MLTextClassifier per label set (LLM0, LLM3, LLM4).
//  Run once per label set, changing datasetName. Paths are relative to the release root.
//

import CreateML
import Foundation
import TabularData
import NaturalLanguage

print("▶️ Program started")

do {
    // ── Change ONLY this filename between variant runs ──────────────
    let datasetName = "LLM4"   // "LLM0" | "LLM3" | "LLM4"
    // ────────────────────────────────────────────────────────────────
    let dataDir = "data/labels"
    let modelDir = "models"
    let dataURL = URL(fileURLWithPath: "\(dataDir)/\(datasetName).csv")

    let data = try DataFrame(contentsOfCSVFile: dataURL)
    print("Dataset: \(datasetName)  |  Loaded \(data.rows.count) rows")

    let (trainingData, testingData) = data.stratifiedSplit(on: "label", by: 0.8)
    print("Split: \(trainingData.rows.count) train / \(testingData.rows.count) test")

    let parameters = MLTextClassifier.ModelParameters(
        validation: .split(strategy: .automatic),
        language: .french
    )

    let classifier = try MLTextClassifier(
        trainingData: trainingData,
        textColumn: "text",
        labelColumn: "label",
        parameters: parameters
    )

    let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
    let validationAccuracy = (1.0 - classifier.validationMetrics.classificationError) * 100
    print("Training: \(trainingAccuracy)%  Validation: \(validationAccuracy)%")

    let evaluation = classifier.evaluation(on: testingData,
                                           textColumn: "text",
                                           labelColumn: "label")
    print("Test accuracy: \((1.0 - evaluation.classificationError) * 100)%")

    // ── Confusion matrix + P/R/F1 (Harassment = positive class) ────
    let texts = testingData["text",  String.self].map { $0 ?? "" }
    let golds = testingData["label", String.self].map { $0 ?? "" }
    var tp = 0, fp = 0, fn = 0, tn = 0
    for (text, gold) in zip(texts, golds) {
        let pred = try classifier.prediction(from: text)
        switch (pred, gold) {
        case ("Harassment", "Harassment"): tp += 1
        case ("Harassment", "Neutral"):    fp += 1
        case ("Neutral",    "Harassment"): fn += 1
        default:                           tn += 1
        }
    }
    let precision = Double(tp) / Double(max(tp + fp, 1))
    let recall    = Double(tp) / Double(max(tp + fn, 1))
    let f1        = (precision + recall) > 0
                  ? 2 * precision * recall / (precision + recall) : 0
    print(String(format: "Confusion [TP FP; FN TN]: [%d %d; %d %d]", tp, fp, fn, tn))
    print(String(format: "Precision: %.4f  Recall: %.4f  F1: %.4f", precision, recall, f1))
    // ────────────────────────────────────────────────────────────────

    let metadata = MLModelMetadata(author: "Anonymous",
                                   shortDescription: "French harassment classifier (\(datasetName))",
                                   version: "1.0")
    try classifier.write(
        to: URL(fileURLWithPath: "\(modelDir)/FrenchClassifier_\(datasetName).mlmodel"),
        metadata: metadata
    )
    print("Model saved ✅  FrenchClassifier_\(datasetName).mlmodel")
} catch {
    print("Error: \(error)")
}
