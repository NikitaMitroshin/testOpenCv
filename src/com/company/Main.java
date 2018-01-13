package com.company;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

public class Main {

    private static final String FILE_NAME = "resources/images/test/white_face.jpg";
    private static final String RESULT_FILE_NAME = "resources/images/result/result%s.jpg";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        final Mat radImage = Imgcodecs.imread(FILE_NAME);

        MatOfRect faces = new MatOfRect();
        Mat grayFrame = new Mat();

        // convert the frame in gray scale
        Imgproc.cvtColor(radImage, grayFrame, Imgproc.COLOR_BGR2GRAY);
        // equalize the frame histogram to improve the result
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // compute minimum face size (20% of the frame height, in our case)
        int absoluteFaceSize = 0;
        int height = grayFrame.rows();
        if (Math.round(height * 0.1f) > 0) {
            absoluteFaceSize = Math.round(height * 0.1f);
        }


        final CascadeClassifier faceCascade = new CascadeClassifier();
        faceCascade.load("resources/haarcascades/haarcascade_frontalface_alt.xml");
        // detect faces
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, Objdetect.CASCADE_SCALE_IMAGE,
                new Size(absoluteFaceSize, absoluteFaceSize), new Size());

        // each rectangle in faces is a face: draw them!
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
//        for (Rect aFacesArray : facesArray) {
//            Imgproc.rectangle(radImage, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
            Mat image_roi = new Mat(radImage,facesArray[i]);
            Imgcodecs.imwrite(String.format(RESULT_FILE_NAME, System.currentTimeMillis()), image_roi);
        }

//        Imgcodecs.imwrite(RESULT_FILE_NAME, radImage);
    }
}
