package com.example.arm64opencvfacedetection

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Button
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private lateinit var buttonStartPreview: Button
    private lateinit var buttonStopPreview: Button

    private lateinit var openCvCameraView: CameraBridgeViewBase

    private var isPreviewActive = false
    private var isOpenCvInitialized = false

    private val cameraPermissionRequestCode = 100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        buttonStartPreview = findViewById(R.id.buttonStartPreview)
        buttonStopPreview = findViewById(R.id.buttonStopPreview)

        openCvCameraView = findViewById(R.id.cameraView)

        isOpenCvInitialized = OpenCVLoader.initLocal()

        // Request access to camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            // Permission is not granted, request it
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                cameraPermissionRequestCode
            )
        }

        openCvCameraView.setCvCameraViewListener(this)
        openCvCameraView.setCameraIndex(1)

        buttonStartPreview.setOnClickListener {
            openCvCameraView.setCameraPermissionGranted()
            openCvCameraView.enableView()

            updateControls()
        }

        buttonStopPreview.setOnClickListener {
            openCvCameraView.disableView()

            updateControls()
        }

        updateControls()

        loadHaarCascade()
    }

    private fun updateControls() {
        if(!isOpenCvInitialized) {
            buttonStartPreview.isEnabled = false
            buttonStopPreview.isEnabled = false
        } else {
            buttonStartPreview.isEnabled = !isPreviewActive
            buttonStopPreview.isEnabled = isPreviewActive
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        isPreviewActive = true
    }

    override fun onCameraViewStopped() {
        isPreviewActive = false
    }

    private lateinit var frame: Mat
    private lateinit var grayFrame: Mat
    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        if (!::frame.isInitialized) {
            frame = Mat(inputFrame?.rgba()?.size(), CvType.CV_8UC4)
            grayFrame = Mat(inputFrame?.rgba()?.size(), CvType.CV_8UC1)
        } else {
            frame = inputFrame?.rgba() ?: return Mat()
            grayFrame = Mat(inputFrame.rgba()?.size(), CvType.CV_8UC1)
        }

        try {
            // Convert to grayscale
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.equalizeHist(grayFrame, grayFrame)

            val faces = MatOfRect()
            faceCascade.detectMultiScale(grayFrame, faces)

            val faceArray = faces.toArray()
            if (faceArray.isNotEmpty()) {
                val rect = faceArray[0]
                Imgproc.rectangle(frame,
                    Point(rect.x.toDouble(), rect.y.toDouble()),
                    Point(rect.x + rect.width.toDouble(), rect.y + rect.height.toDouble()),
                    Scalar(0.0, 255.0, 0.0),
                    2)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error during face detection: ${e.message}")
        }

        return frame
    }

    private lateinit var faceCascade: CascadeClassifier

    private fun loadHaarCascade() {
        try {
            val cascadeFile = getPath("haarcascade_frontalface_default.xml", this)
            faceCascade = CascadeClassifier(cascadeFile)
            if (faceCascade.empty()) {
                Log.e("MainActivity", "Failed to load cascade classifier")
            } else {
                Log.d("MainActivity", "Loaded cascade classifier from $cascadeFile")
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "Error loading cascade classifier: ${e.message}")
        }
    }

    @Throws(IOException::class)
    private fun getPath(file: String, context: Context): String {
        val assetManager = context.assets
        val inputStream: InputStream = assetManager.open(file)
        val outFile = File(context.filesDir, file)
        val outputStream = FileOutputStream(outFile)
        val buffer = ByteArray(1024)
        var read: Int
        while (inputStream.read(buffer).also { read = it } != -1) {
            outputStream.write(buffer, 0, read)
        }
        inputStream.close()
        outputStream.close()
        return outFile.absolutePath
    }
}