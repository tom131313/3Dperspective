package app;

import java.util.Arrays;
import java.util.stream.Collectors;

import org.opencv.highgui.HighGui;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Point;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class App {
    static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // Load the native library.
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
        System.out.println("P " + P.dump());
        System.out.println("T " + T.dump());
        System.out.println("Rphi " + Rphi.dump());
        System.out.println("Rtheta " + Rtheta.dump());
        System.out.println("Rgamma " + Rgamma.dump());
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
    
        System.out.println("ptsInMat " + ptsInMat + " " + ptsInMat.dump());
        System.out.println("F " + F + " " + F.dump());
        Core.perspectiveTransform(ptsInMat, ptsOutMat, F);//Transform points
        System.out.println("ptsOutMat " + ptsOutMat + " " + ptsOutMat.dump());
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
           System.out.println("ptsOutPt2f " + ptsOutPt2f[i]);
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

        System.out.println("ptsInPt2fTemp " + ptsInPt2fTemp.dump());
        System.out.println("ptsOutPt2fTemp " + ptsOutPt2fTemp.dump());
        Mat warp=Imgproc.getPerspectiveTransform(ptsInPt2fTemp, ptsOutPt2fTemp);
        warp.copyTo(M);
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
        System.out.println("corners " + corners + " " + corners.dump());
    }
    
    static void warpImage(Mat src,
                   double    theta,
                   double    phi,
                   double    gamma,
                   double    scale,
                   double    fovy,
                   Mat      dst,
                   Mat      M,
                   MatOfPoint2f corners){
        double halfFovy=fovy*0.5;
        double d=Math.hypot(src.cols(),src.rows());
        double sideLength=scale*d/Math.cos(Math.toRadians(halfFovy));
        System.out.println("d " + d + ", sideLength " + sideLength);
        warpMatrix(src.size(), theta, phi, gamma, scale, fovy, M, corners);//Compute warp matrix
        System.out.println("M " + M + " " + M.dump());
        Imgproc.warpPerspective(src, dst, M, new Size(sideLength,sideLength));//Do actual image warp
    }

    public static void main(String[] args)
    {
        int c = -1;
        Mat m = new Mat();
        Mat disp = new Mat();
        Mat warp = new Mat();
        MatOfPoint2f corners = new MatOfPoint2f(new Point(0,0),new Point(0,0),new Point(0,0),new Point(0,0));

        String filename = "lena.jpg";
        m = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
        if (m.empty()) {
            System.out.println("Error opening image");
            System.exit(-1);
        }

        double scale = 1.;
        double fovy = 53.;
        double halfFovy=fovy*0.5;

        VideoCapture cap;
        cap = new VideoCapture();
        cap.open(1);
        cap.read(m);
        warpImage(m, 5, 50, 0, 1, 30, disp, warp, corners); // fovy = rad2deg(arctan2(640,480)) = 53 ??

        while(c == -1 && cap.isOpened()) {
            cap.read(m);
            double d=Math.hypot(m.cols(),m.rows());
            double sideLength=scale*d/Math.cos(Math.toRadians(halfFovy));
            Imgproc.warpPerspective(m, disp, warp, new Size(sideLength,sideLength));//Do actual image warp
            HighGui.imshow("Disp", disp);
            HighGui.imshow("Orig", m);
            c = HighGui.waitKey(25);
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
