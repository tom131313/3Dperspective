// 3D rotation of an image file or camera stream

// sliders (trackbars) used to vary the rotation angles and Field of View

// close a window or press a key in an image window to terminate the program

// Combination of a Java conversion of a StackOverflow 3D rotation answer
// and the OpenCV trackbar example

// https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles

// https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html

package app;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.awt.BorderLayout;
import java.awt.Container;
import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.core.MatOfPoint2f;

public class Rotate {
    static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // Load the native library.
    }

    private final int ANGLE_SLIDER_MIN = -180;
    private final int ANGLE_SLIDER_MAX = 180;
    private JFrame frame;   
    static JLabel labelX = new JLabel("X");
    static JLabel labelY = new JLabel("Y");
    static JLabel labelZ = new JLabel("Z");
    static JLabel labelFOVY = new JLabel("FOV Y");

    static AtomicBoolean recalculate = new AtomicBoolean(true);
    static AtomicInteger angleZ = new AtomicInteger(5);
    static AtomicInteger angleX = new AtomicInteger(50);
    static AtomicInteger angleY = new AtomicInteger(0);
    static AtomicInteger scale = new AtomicInteger(1);
    static AtomicInteger fovy = new AtomicInteger(53);

        public Rotate(String[] args) {
            // Create and set up the window.
            frame = new JFrame("Angle Control");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // end program if frame closed
            // Set up the content pane.
            addComponentsToPane(frame.getContentPane());
            // Use the content pane's default BorderLayout. No need for
            // setLayout(new BorderLayout());
            // Display the window.
            frame.pack();
            frame.setVisible(true);
        }
    
        /**
         * JFrame
         *  components of JFrame container
         *   JPanel
         *     JLabel (string text)
         *     JSlider
         *       changeListener
         *         stateChanged
         */
        private void addComponentsToPane(Container pane) {
            if (!(pane.getLayout() instanceof BorderLayout)) {
                pane.add(new JLabel("Container doesn't use BorderLayout!"));
                return;
            }
    
                JPanel sliderPanel = new JPanel();
                sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS));
    
                labelX.setText(String.format("X %d", angleX.get()));
                sliderPanel.add(labelX);
    
                JSlider sliderX = new JSlider(ANGLE_SLIDER_MIN, ANGLE_SLIDER_MAX, angleX.get());
                sliderX.setMajorTickSpacing(90);
                sliderX.setMinorTickSpacing(5);
                sliderX.setPaintTicks(true);
                sliderX.setPaintLabels(true);
                sliderX.addChangeListener(new ChangeListener() {
                    @Override
                    public void stateChanged(ChangeEvent e) {
                        JSlider source = (JSlider) e.getSource();
                        angleX.set(source.getValue());
                        updateX();
                    }
                });
                sliderPanel.add(sliderX);
    
                labelY.setText(String.format("Y %d", angleY.get()));
                sliderPanel.add(labelY);
    
                JSlider sliderY = new JSlider(ANGLE_SLIDER_MIN, ANGLE_SLIDER_MAX, angleY.get());
                sliderY.setMajorTickSpacing(90);
                sliderY.setMinorTickSpacing(5);
                sliderY.setPaintTicks(true);
                sliderY.setPaintLabels(true);
                sliderY.addChangeListener(new ChangeListener() {
                    @Override
                    public void stateChanged(ChangeEvent e) {
                        JSlider source = (JSlider) e.getSource();
                        angleY.set(source.getValue());
                        updateY();
                    }
                });
                sliderPanel.add(sliderY);
    
                labelZ.setText(String.format("Z %d", angleZ.get()));
                sliderPanel.add(labelZ);
    
                JSlider sliderZ = new JSlider(ANGLE_SLIDER_MIN, ANGLE_SLIDER_MAX, angleZ.get());
                sliderZ.setMajorTickSpacing(90);
                sliderZ.setMinorTickSpacing(5);
                sliderZ.setPaintTicks(true);
                sliderZ.setPaintLabels(true);
                sliderZ.addChangeListener(new ChangeListener() {
                    @Override
                    public void stateChanged(ChangeEvent e) {
                        JSlider source = (JSlider) e.getSource();
                        angleZ.set(source.getValue());
                        updateZ();
                    }
                });
                sliderPanel.add(sliderZ);
    

                labelFOVY.setText(String.format("FOV Y %d", fovy.get()));
                sliderPanel.add( labelFOVY);
    
                JSlider sliderFOVY = new JSlider(1, 179, fovy.get());
                sliderFOVY.setMajorTickSpacing(89);
                sliderFOVY.setMinorTickSpacing(5);
                sliderFOVY.setPaintTicks(true);
                sliderFOVY.setPaintLabels(true);
                sliderFOVY.addChangeListener(new ChangeListener() {
                    @Override
                    public void stateChanged(ChangeEvent e) {
                        JSlider source = (JSlider) e.getSource();
                        fovy.set(source.getValue());
                        updateFOVY();
                    }
                });
                sliderPanel.add(sliderFOVY);
    
                pane.add(sliderPanel, BorderLayout.PAGE_START);                
        }
    
        private void updateX() {
            labelX.setText(String.format("X %d", angleX.get()));
            recalculate.set(true);
         }
    
        private void updateY() {
            labelY.setText(String.format("Y %d", angleY.get()));
            recalculate.set(true);
        }
    
        private void updateZ() {
            labelZ.setText(String.format("Z %d", angleZ.get()));
            recalculate.set(true);
        }
    
        private void updateFOVY() {
            labelFOVY.setText(String.format("FOV Y %d", fovy.get()));
            recalculate.set(true);
        }

    static void warpMatrix(Size   sz,
                    double theta,
                    double phi,
                    double gamma,
                    double scale,
                    double fovy,
                    Mat   M,
                    MatOfPoint2f corners) {

        double st=Math.sin(Math.toRadians(theta));
        double ct=Math.cos(Math.toRadians(theta));
        double sp=Math.sin(Math.toRadians(phi));
        double cp=Math.cos(Math.toRadians(phi));
        double sg=Math.sin(Math.toRadians(gamma));
        double cg=Math.cos(Math.toRadians(gamma));
    
        double halfFovy=fovy*0.5;
        double d=Math.hypot(sz.width,sz.height);
        double sideLength=scale*d/Math.cos(Math.toRadians(halfFovy));
        double h=d/(2.0*Math.sin(Math.toRadians(halfFovy)));
        double n=h-(d/2.0);
        double f=h+(d/2.0);
    
        Mat F=new Mat(4,4, CvType.CV_64FC1);//Allocate 4x4 transformation matrix F
        Mat Rtheta=Mat.eye(4,4, CvType.CV_64FC1);//Allocate 4x4 rotation matrix around Z-axis by theta degrees
        Mat Rphi=Mat.eye(4,4, CvType.CV_64FC1);//Allocate 4x4 rotation matrix around X-axis by phi degrees
        Mat Rgamma=Mat.eye(4,4, CvType.CV_64FC1);//Allocate 4x4 rotation matrix around Y-axis by gamma degrees
    
        Mat T=Mat.eye(4,4, CvType.CV_64FC1);//Allocate 4x4 translation matrix along Z-axis by -h units
        Mat P=Mat.zeros(4,4, CvType.CV_64FC1);//Allocate 4x4 projection matrix
                                                // zeros instead of eye as in github manisoftwartist/perspectiveproj
    
        //Rtheta Z
        Rtheta.put(0,0, ct);
        Rtheta.put(1,1, ct);
        Rtheta.put(0,1, -st);
        Rtheta.put(1,0, st);
        //Rphi X
        Rphi.put(1,1, cp);
        Rphi.put(2,2, cp);
        Rphi.put(1,2, -sp);
        Rphi.put(2,1, sp);
        //Rgamma Y
        Rgamma.put(0,0, cg);
        Rgamma.put(2,2, cg);
        Rgamma.put(0,2, -sg); // sign reversed? Math different convention than computer graphics according to Wikipedia
        Rgamma.put(2,0, sg);
        //T
        T.put(2,3, -h);
        //P Perspective Matrix (see also in computer vision a camera matrix or (camera) projection matrix is a 3x4 matrix which describes the mapping of a pinhole camera from 3D points in the world to 2D points in an image.)
        P.put(0,0, 1.0/Math.tan(Math.toRadians(halfFovy)));
        P.put(1,1, 1.0/Math.tan(Math.toRadians(halfFovy)));
        P.put(2,2, -(f+n)/(f-n));
        P.put(2,3, -(2.0*f*n)/(f-n));
        P.put(3,2, -1.0);
        System.out.println("P\n" + P.dump());
        System.out.println("T\n" + T.dump());
        System.out.println("Rphi\n" + Rphi.dump());
        System.out.println("Rtheta\n" + Rtheta.dump());
        System.out.println("Rgamma\n" + Rgamma.dump());
        //Compose transformations
        //F=P*T*Rphi*Rtheta*Rgamma;//Matrix-multiply to produce master matrix
        //gemm(Mat src1, Mat src2, double alpha, Mat src3, double beta, Mat dst)
        //dst = alpha*src1.t()*src2 + beta*src3.t(); // w or w/o the .t() transpose
        // D=α∗AB+β∗C
        Mat F1 = new Mat();
        Mat F2 = new Mat();
        Mat F3 = new Mat();
        Core.gemm(P, T, 1, new Mat(), 0, F1);
        Core.gemm(F1, Rphi, 1, new Mat(), 0, F2);
        Core.gemm(F2, Rtheta, 1, new Mat(), 0, F3);
        Core.gemm(F3, Rgamma, 1, new Mat(), 0, F);
        P.release();
        T.release();
        Rphi.release();
        Rtheta.release();
        Rgamma.release();
        F1.release();
        F2.release();
        F3.release();

        //Transform 4x4 points
        double[] ptsIn = new double[4*3];
        double[] ptsOut = new double[4*3];
        double halfW=sz.width/2, halfH=sz.height/2;
    
        ptsIn[0]=-halfW;ptsIn[ 1]= halfH;
        ptsIn[3]= halfW;ptsIn[ 4]= halfH;
        ptsIn[6]= halfW;ptsIn[ 7]=-halfH;
        ptsIn[9]=-halfW;ptsIn[10]=-halfH;
        ptsIn[2]=ptsIn[5]=ptsIn[8]=ptsIn[11]=0;//Set Z component to zero for all 4 components
    
        Mat ptsInMat = new Mat(1,4,CvType.CV_64FC3);
        ptsInMat.put(0,0, ptsIn);

        Mat ptsOutMat = new Mat(1,4,CvType.CV_64FC3);
    
        System.out.println("ptsInMat " + ptsInMat + "\n" + ptsInMat.dump());
        System.out.println("F " + F + "\n" + F.dump());
        Core.perspectiveTransform(ptsInMat, ptsOutMat, F);//Transform points
        System.out.println("ptsOutMat " + ptsOutMat + "\n" + ptsOutMat.dump());
        ptsInMat.release();
        F.release();
        ptsOutMat.get(0, 0, ptsOut);
        ptsOutMat.release();
        System.out.println(toString(ptsOut));
        System.out.println(halfW + " " + halfH);

        //Get 3x3 transform and warp image
        Point[] ptsInPt2f = new Point[4];
        Point[] ptsOutPt2f = new Point[4];
        for(int i=0;i<4;i++){
            ptsInPt2f[i] = new Point(0, 0);
            ptsOutPt2f[i] = new Point(0, 0);
            System.out.println(i);
            System.out.println("points " + ptsIn [i*3+0] + " " + ptsIn [i*3+1]);
            Point ptIn = new Point(ptsIn [i*3+0], ptsIn [i*3+1]);
            Point ptOut = new Point(ptsOut[i*3+0], ptsOut[i*3+1]);

            ptsInPt2f[i].x  = ptIn.x+halfW;
            ptsInPt2f[i].y  = ptIn.y+halfH;

            ptsOutPt2f[i].x = (ptOut.x+1) * sideLength*0.5;
            ptsOutPt2f[i].y = (ptOut.y+1) * sideLength*0.5;
           System.out.println("ptsOutPt2f\n" + ptsOutPt2f[i]);
        }  

        Mat ptsInPt2fTemp =  Mat.zeros(4,1,CvType.CV_32FC2);
        ptsInPt2fTemp.put(0, 0,
            ptsInPt2f[0].x,ptsInPt2f[0].y,
            ptsInPt2f[1].x,ptsInPt2f[1].y,
            ptsInPt2f[2].x,ptsInPt2f[2].y,
            ptsInPt2f[3].x,ptsInPt2f[3].y);

        Mat ptsOutPt2fTemp = Mat.zeros(4,1,CvType.CV_32FC2);
        ptsOutPt2fTemp.put(0, 0,
            ptsOutPt2f[0].x,ptsOutPt2f[0].y,
            ptsOutPt2f[1].x,ptsOutPt2f[1].y,
            ptsOutPt2f[2].x,ptsOutPt2f[2].y,
            ptsOutPt2f[3].x,ptsOutPt2f[3].y);

        System.out.println("ptsInPt2fTemp\n" + ptsInPt2fTemp.dump());
        System.out.println("ptsOutPt2fTemp\n" + ptsOutPt2fTemp.dump());
        Mat warp=Imgproc.getPerspectiveTransform(ptsInPt2fTemp, ptsOutPt2fTemp);
        warp.copyTo(M); // return the warp matrix through the parameter list
        ptsInPt2fTemp.release();
        warp.release();

        //Load corners vector
        if(corners != null)
        {
            corners.put(0,0, ptsOutPt2f[0].x, ptsOutPt2f[0].y//Push Top Left corner
            , ptsOutPt2f[1].x, ptsOutPt2f[1].y//Push Top Right corner
            , ptsOutPt2f[2].x, ptsOutPt2f[2].y//Push Bottom Right corner
            , ptsOutPt2f[3].x, ptsOutPt2f[3].y);//Push Bottom Left corner
        }
        ptsOutPt2fTemp.release();
        System.out.println("corners " + corners + "\n" + corners.dump());
    }
    
    static void warpImage(Mat src,
                   double    theta, //z
                   double    phi, //x
                   double    gamma, //y
                   double    scale,
                   double    fovy, //field of view y
                   Mat      dst,
                   Mat      M,
                   MatOfPoint2f corners){
        double halfFovy=fovy*0.5;
        double d=Math.hypot(src.cols(),src.rows());
        double sideLength=scale*d/Math.cos(Math.toRadians(halfFovy));
        System.out.println("d " + d + ", sideLength " + sideLength);
        warpMatrix(src.size(), theta, phi, gamma, scale, fovy, M, corners);//Compute warp matrix
        System.out.println("M " + M + "\n" + M.dump());
        Imgproc.warpPerspective(src, dst, M, new Size(sideLength,sideLength));//Do actual image warp
    }

 
    public static void main(String[] args)
    {
        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI - the slider bars (not the OpenCV images).
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new Rotate(args);
            }
        });
 
        int c = -1;
        Mat m = new Mat();
        Mat disp = new Mat();
        Mat warp = new Mat();
        MatOfPoint2f corners = new MatOfPoint2f(new Point(0,0),new Point(0,0),new Point(0,0),new Point(0,0));

        double halfFovy=fovy.get()*0.5;

        // checks for the speciified camera and uses it if present.
        // If that camera isn't available, them open the static image file.

        int camera = 1; //0 internal, 1 is external

        VideoCapture cap;
        cap = new VideoCapture();
        cap.open(camera);

        if( cap.isOpened() ) {
            cap.read(m);
        }
        else {
            System.out.println("No camera");

            System.out.println("Opening image file");

            String imagePath = "lena.jpg";
            if (args.length > 0) {
                imagePath = args[0];
            }

            m = Imgcodecs.imread(imagePath);

            if (m.empty()) {
                System.out.println("Empty image: " + imagePath);
                System.exit(-1);
            }
        }

        // outer loop runs until a key is presses in the OpenCV image screens
        // inner loop runs until an update event for a slider bar is generated
        loop:
        while( true ) {
            // new value from a slider bar event so calculate a new wrap matrix
            warpImage(m, angleZ.get(), angleX.get(), angleY.get(), 1, fovy.get(), disp, warp, corners); // fovy = rad2deg(arctan2(640,480)) = 53 ??
            recalculate.set(false);
             
        while( ! recalculate.get() ) {
            // use the warp matrix to wrap each image (still or camera frame)
            if (cap.isOpened()) cap.read(m);
            double d=Math.hypot(m.cols(),m.rows());
            double sideLength=scale.get()*d/Math.cos(Math.toRadians(halfFovy));
            Imgproc.warpPerspective(m, disp, warp, new Size(sideLength,sideLength));//Do actual image warp
            HighGui.imshow("Disp", disp);
            HighGui.imshow("Orig", m);
            c = HighGui.waitKey(25); // wait so not beating on computer, 25 millisecs is about 40 fps
            if (c != -1) break loop; // end program if key is pressed in an OpenCV window

        }
        }

        m.release();
        disp.release();
        warp.release();
        corners.release();
        System.exit(0);
    }

    static String toString(double[] array) {
        return Arrays.stream(array)
                .mapToObj(i -> String.format("%5.2f", i))
                .collect(Collectors.joining(", ", "[", "]"));
                //.collect(Collectors.joining("|", "|", "|"));
            }
}
