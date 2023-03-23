//
//  PhotoViewController.swift
//  Object Classification Demo
//
//  Created by Abraham on 3/20/23.
//

import UIKit
import PhotosUI
import CoreML
import Vision

class PhotoViewController: UIViewController, PHPickerViewControllerDelegate {
    // UI view and label from storyboard scene
    @IBOutlet weak var sampleImageView: UIImageView!
    @IBOutlet weak var predictionLabel: UILabel!
    
    // UI image object used for analysis
    var sampleImage: UIImage!
    
    private static let imageClassifier = createImageClassifier()
    
    // Prediction structure to retrieve results
    struct Prediction {
        // Name of object recognized
        let classification: String
        // percentage probability that image represents said object, according to model
        let confidencePercentage: String
    }
    
    // Prediction Handlers used to perform classification task
    typealias ImagePredictionHandler = (_ predictions: [Prediction]?) -> Void
    private var predictionHandlers = [VNRequest: ImagePredictionHandler]()

    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    
    // Initilizes classification model for usage, based on pretrained or custom model
    static func createImageClassifier() -> VNCoreMLModel {
        // Use a default model configuration.
        let defaultConfig = MLModelConfiguration()

        // Create an instance of the image classifier's wrapper class.
        let imageClassifierWrapper = try? MobileNetV2(configuration: defaultConfig)

        guard let imageClassifier = imageClassifierWrapper else {
            fatalError("App failed to create an image classifier model instance.")
        }

        // Get the underlying model instance.
        let imageClassifierModel = imageClassifier.model

        // Create a Vision instance using the image classifier's model instance.
        guard let imageClassifierVisionModel = try? VNCoreMLModel(for: imageClassifierModel) else {
            fatalError("App failed to create a `VNCoreMLModel` instance.")
        }

        return imageClassifierVisionModel
    }
    
    // Upload image button action
    @IBAction func onUploadTapped(_ sender: UIButton) {
        if PHPhotoLibrary.authorizationStatus(for: .readWrite) != .authorized {
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { [weak self] status in
                switch status {
                case .authorized:
                    DispatchQueue.main.async {
                        self?.presentImagePicker()
                    }
                default:
                    DispatchQueue.main.async {
                        self?.presentGoToSettingsAlert()
                    }
                }
            }
        } else {
            presentImagePicker()
        }
    }
    
    private func visionRequestHandler(_ request: VNRequest, error: Error?) {
        // Remove the caller's handler from the dictionary and keep a reference to it.
        guard let predictionHandler = predictionHandlers.removeValue(forKey: request) else {
            fatalError("Every request must have a prediction handler.")
        }

        // Start with a `nil` value in case there's a problem.
        var predictions: [Prediction]? = nil

        // Call the client's completion handler after the method returns.
        defer {
            // Send the predictions back to the client.
            predictionHandler(predictions)
        }

        // Check for an error first.
        if let error = error {
            print("Vision image classification error...\n\n\(error.localizedDescription)")
            return
        }

        // Check that the results aren't `nil`.
        if request.results == nil {
            print("Vision request had no results.")
            return
        }

        // Cast the request's results as an `VNClassificationObservation` array.
        guard let observations = request.results as? [VNClassificationObservation] else {
            // Image classifiers, like MobileNet, only produce classification observations.
            // However, other Core ML model types can produce other observations.
            // Make sure to be familiar with model used
            print("VNRequest produced the wrong result type: \(type(of: request.results)).")
            return
        }

        // Create a prediction array from the observations.
        predictions = observations.map { observation in
            // Convert each observation into an `ImagePredictor.Prediction` instance.
            Prediction(classification: observation.identifier,
                       confidencePercentage: observation.confidencePercentageString)
        }
    }
    
    // Formats predictions into strings for usage (UI display)
    private func imagePredictionHandler(_ predictions: [PhotoViewController.Prediction]?) {
        guard let predictions = predictions else {
            updatePredictionLabel("No predictions. (Check console log.)")
            return
        }
        let formattedPredictions = formatPredictions(predictions)
        let predictionString = formattedPredictions.joined(separator: "\n")
        updatePredictionLabel(predictionString)
    }
    
    private func formatPredictions(_ predictions: [PhotoViewController.Prediction]) -> [String] {
        // Vision sorts the classifications in descending confidence order.
        
        // let predictionsToShow = 2;   // This variable allows x number of predictions to be shown.
                                                    // Used within .prefix()
        let topPredictions: [String] = predictions.prefix(2).map { prediction in
            var name = prediction.classification
            if let firstComma = name.firstIndex(of: ",") {
                name = String(name.prefix(upTo: firstComma))
            }
            return "\(name) - \(prediction.confidencePercentage)%"
        }
        return topPredictions
    }
    // Uploads label in UI storyboard
    func updatePredictionLabel(_ message: String) {
        DispatchQueue.main.async {
            self.predictionLabel.text = message
        }
    }
    
    // Generates a new request instance that uses the Image Predictor's image classifier model.
    private func createImageClassificationRequest() -> VNImageBasedRequest {
        // Create an image classification request with an image classifier model.
        let imageClassificationRequest = VNCoreMLRequest(model: PhotoViewController.imageClassifier, completionHandler: visionRequestHandler)
        // Performs image cropping before analysis
        imageClassificationRequest.imageCropAndScaleOption = .centerCrop
        return imageClassificationRequest
    }
    
}

// Extension with Image picker methods
extension PhotoViewController {
    private func presentImagePicker() {
        var config = PHPickerConfiguration(photoLibrary: PHPhotoLibrary.shared())
        config.filter = .images
        config.preferredAssetRepresentationMode = .current
        config.selectionLimit = 1
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = self
        present(picker, animated: true)
    }
    
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        let result = results.first
        guard let provider = result?.itemProvider, provider.canLoadObject(ofClass: UIImage.self) else { return }
            provider.loadObject(ofClass: UIImage.self) { [weak self] object, error in
                guard let image = object as? UIImage else {
                    self?.showAlert()
                    return
                }

                if let error = error {
                  DispatchQueue.main.async { [weak self] in self?.showAlert(for:error) }
                
                } else {
                    DispatchQueue.main.async {
                        // Set image on preview image view
                        self?.sampleImageView.image = image
                        // Sets main image object to be classified
                        self?.sampleImage = image
                    }
                }
                // Checks if object is UIImage
                guard object is UIImage else { return }
                    
                print("🌉 We have an image!")
                                    
                // Create an image classification request with an image classifier model.
                typealias VNRequestCompletionHandler = (VNRequest, Error?) -> Void
                    
                let imageClassificationRequest = self!.createImageClassificationRequest()
                // initializes handlers
                self?.predictionHandlers[imageClassificationRequest] = self?.imagePredictionHandler

                // Converts UIImage from picker to CGImage for analysis
                /// UIImage is a high level image representation used for displaying and manipulating images
                /// CGImage is a low level image representation, allowing more control of image data for processing.
                guard let photo = self?.sampleImage.cgImage else {
                    fatalError("Photo doesn't have underlying CGImage.")
                }
                let handler = VNImageRequestHandler(cgImage: photo, orientation: CGImagePropertyOrientation.up)
                // initializes requests
                let requests: [VNRequest] = [imageClassificationRequest]
                // Perform the image classification request.
                try? handler.perform(requests)
        }
    }
    
    private func showAlert(for error: Error? = nil) {
        let alertController = UIAlertController(
            title: "Oops...",
            message: "\(error?.localizedDescription ?? "Please try again...")",
            preferredStyle: .alert)

        let action = UIAlertAction(title: "OK", style: .default)
        alertController.addAction(action)

        present(alertController, animated: true)
    }
    
    func presentGoToSettingsAlert() {
        let alertController = UIAlertController (
            title: "Photo Access Required",
            message: "In order to post a photo to complete a task, we need access to your photo library. You can allow access in Settings",
            preferredStyle: .alert)

        let settingsAction = UIAlertAction(title: "Settings", style: .default) { _ in
            guard let settingsUrl = URL(string: UIApplication.openSettingsURLString) else { return }

            if UIApplication.shared.canOpenURL(settingsUrl) {
                UIApplication.shared.open(settingsUrl)
            }
        }

        alertController.addAction(settingsAction)
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        alertController.addAction(cancelAction)

        present(alertController, animated: true, completion: nil)
    }
}

