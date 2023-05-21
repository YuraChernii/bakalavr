import UIKit
import AVKit
import Vision
import AVFoundation
import Alamofire

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    let identifierLabel: UILabel = {
        let label = UILabel()
        label.backgroundColor = UIColor.black.withAlphaComponent(0.5) // Semi-transparent black background
        label.textColor = .white // White text color
        label.font = UIFont.boldSystemFont(ofSize: 24) // Bold font with size 24
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        //guard let captureDevice = AVCaptureDevice.default(for: .video) else { return }
        // Use the front camera
        guard let captureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else { return }

        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        captureSession.addInput(input)
        
        captureSession.startRunning()
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(dataOutput)
        
        
//        VNImageRequestHandler(cgImage: <#T##CGImage#>, options: [:]).perform(<#T##requests: [VNRequest]##[VNRequest]#>)
        
        setupIdentifierConfidenceLabel()
    }
    
    fileprivate func setupIdentifierConfidenceLabel() {
        view.addSubview(identifierLabel)
        identifierLabel.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -32).isActive = true
        identifierLabel.leftAnchor.constraint(equalTo: view.leftAnchor).isActive = true
        identifierLabel.rightAnchor.constraint(equalTo: view.rightAnchor).isActive = true
        identifierLabel.heightAnchor.constraint(equalToConstant: 50).isActive = true
    }
    var frameCounter = 0
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameCounter += 1

        // Process only every 5 frames
        if frameCounter % 10 != 0 {
            return
        }
        
        print("Camera was able to capture a frame:", Date())
        
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                let context = CIContext(options: nil)
                guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
                guard let imageData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.9) else { return }
                
                let serverURL = "http://192.168.50.219:5000/predict"
        AF.upload(multipartFormData: { multipartFormData in
            multipartFormData.append(imageData, withName: "image", fileName: "file.jpg", mimeType: "image/jpeg")
            // Add any additional form data here
            // For example, a form field 'name':
            // if let nameData = "John Doe".data(using: .utf8) {
            //     multipartFormData.append(nameData, withName: "name")
            // }
        }, to: serverURL)
        .responseJSON { response in
            print(response)
            
            if let result = response.value as? [[Double]],
               let innerArray = result.first,
               let value = innerArray.first {
                if value > 0.5 {
                    DispatchQueue.main.async {
                        self.identifierLabel.text = "Wakefulness detected. - \((value * 100).rounded() / 100)"
                    }
                } else {
                    DispatchQueue.main.async {
                        self.identifierLabel.text = "Wake up!!! - \((value * 100).rounded() / 100)"
                    }
                }
            } else {
                DispatchQueue.main.async {
                    self.identifierLabel.text = "No face."
                }
            }
        }
        
    }

}
