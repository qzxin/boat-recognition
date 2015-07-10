/***********************************************************************
船只类型识别程序：
功能：通过识别船只图片判断出属于哪一种船型
（只针对集装箱船，沙船，油船，自卸货船）
程序： main.cpp
开发软件：VS2012 , OpenCV 2.4.4
作者：quinn
日期：2014.05.01
***********************************************************************/
#include "cv.h"
#include "highgui.h"
#include<iostream>
#include <fstream>
#include<string>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define FEATURE_NUM 5 //采用的特征数量

CvSeq *GetAreaMaxContour(CvSeq *contour);//在给定的contour中找到面积最大的一个轮廓，并返回指向该轮廓的指针
float getlables(string imgPath);//训练时，根据船只类型添加标签	0:集装箱船 1:沙船 2：油船 3:自卸货船
int getFeatureData(int i,char *path,float trainingData[][FEATURE_NUM]);//提取特征数据函数

int main( int argc, char** argv )
{
	IplImage *train;
	string buf;
	vector<string> img_train_path;
	ifstream img_train( "../SVM_Train.txt" ); //加载需要预测的图片集合，这个文本里存放的是图片全路径，不要标签
	while( img_train )
	{
			if( getline( img_train, buf ) )
			{
					img_train_path.push_back( buf );
			}
	}
	img_train.close();
	int nTrainImg=img_train_path.size();	//训练图片的数量，以便后续分配数组空间
	float (*trainingData)[FEATURE_NUM];	 //存储训练得到的特征数据
	trainingData=new float[nTrainImg][FEATURE_NUM];
	float *labels=new float[nTrainImg];
	cout<<"SVM Train:"<<endl;
	cout<<"	 ImagePath	 "<<"	Labels"<<endl;
	//依次加载训练图片，提取特征数据

	for( string::size_type j = 0; j != nTrainImg; j++ )
	{
				train = cvLoadImage( img_train_path[j].c_str(), CV_LOAD_IMAGE_GRAYSCALE );
				if( train == NULL )
				{
						cout<<" can not load the image: "<<img_train_path[j].c_str()<<endl;
						continue;
				}
				labels[j]=getlables(img_train_path[j].c_str());//添加标签
				char *path=const_cast<char*>(img_train_path[j].c_str());

				cout<<img_train_path[j].c_str()<<"	"<< labels[j]<<endl;
				getFeatureData(j,path,trainingData);//训练，提取特征
	}
	//SVM 训练过程
	// step 1:Array -> Matrix
	Mat labelsMat(nTrainImg, 1, CV_32FC1, labels);	//标签数组->矩阵
	Mat trainingDataMat(nTrainImg, FEATURE_NUM, CV_32FC1, trainingData);	 //特征数组->特征矩阵
	// step 2:New a SVM
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	// step 3:Train 
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
	SVM.save("../SVM_Data.xml");
	// step 4	:Predict
	//	string buf;
	vector<string> img_tst_path;
	ifstream img_tst( "../SVM_Test.txt" ); //加载需要预测的图片集合，这个文本里存放的是图片全路径，不要标签
	while( img_tst )
	{
				if( getline( img_tst, buf ) )
				{
						img_tst_path.push_back( buf );
				}
	}
	img_tst.close();
	IplImage *tst;
	int nTestImg=img_tst_path.size();
	float testData[1][FEATURE_NUM];
	ofstream predict_txt( "../SVM_PREDICT.txt" );//把预测结果存储在这个文本中
	cout<<endl<<endl<<"Vessels Detection:"<<endl;
	cout<<"labels: "<<"0:集装箱船 "<<"1:沙船 "<<"2：油船 "<<"3:自卸货船"<<endl;
	predict_txt<<"labels:"<<endl<<"0:集装箱船 "<<"1:沙船 "<<"2：油船 "<<"3:自卸货船"<<endl;
	for( string::size_type j = 0; j != nTestImg; j++ )
	{
				tst = cvLoadImage( img_tst_path[j].c_str(), CV_LOAD_IMAGE_GRAYSCALE );
				if( tst == NULL )
				{
					cout<<" can not load the image: "<<img_tst_path[j].c_str()<<endl;
					continue;
				}

				char *path=const_cast<char*>(img_tst_path[j].c_str());//string->char*
				getFeatureData(0,path,testData);
				Mat testDataMat(1, FEATURE_NUM, CV_32FC1, testData);
				float ret = SVM.predict( testDataMat );
				

				//	printf( line, "%s %d\r\n", img_tst_path[j].c_str(), ret );
				cout<<img_tst_path[j].c_str()<<"	"<<ret<<endl;
				predict_txt<<img_tst_path[j].c_str()<<"	"<<ret<<endl;
	}
	cvReleaseImage(&tst);
	cvWaitKey(0);
	system("pause");
	//	cvDestroyWindow( "Source" );
	//	cvDestroyWindow( "max" );
	//	cvReleaseImage(&dst);
	return 0;
}
float getlables(string imgPath)
{
			if(imgPath.find("集装箱船")!=imgPath.npos)
				return 0;

			else if(imgPath.find("自卸")!=imgPath.npos)
				return 3;

			else if(imgPath.find("沙船")!=imgPath.npos)
				return 1;

			else if(imgPath.find("油船")!=imgPath.npos)
				return 2;

			else
			{
				cout<<"图片路径文件有误！"<<endl;
				return -1;
			}
}
CvSeq *GetAreaMaxContour(CvSeq *contour)
{
		double contour_area_temp=0,contour_area_max=0;
		CvSeq * area_max_contour = 0 ;//指向面积最大的轮廓
		CvSeq* c=0;
		//printf( "Total Contours Detected: %d\n", Nc );
		for(c=contour; c!=NULL; c=c->h_next )
		{//寻找面积最大的轮廓，即循环结束时的area_max_contour
				contour_area_temp = fabs(cvContourArea( c, CV_WHOLE_SEQ )); //获取当前轮廓面积
				if( contour_area_temp > contour_area_max )
				{
						contour_area_max = contour_area_temp; //找到面积最大的轮廓
						area_max_contour = c;//记录面积最大的轮廓
				}
		}
		return area_max_contour;
}


int getFeatureData(int i,char *path,float trainingData[][FEATURE_NUM])
{
			IplImage* src;
			CvMemStorage* storage = cvCreateMemStorage(0);
			CvSeq* contour = 0;
			CvSeq* max_contour = 0;
			CvScalar color = CV_RGB( 0, 255,255 );

			src=cvLoadImage(path,CV_LOAD_IMAGE_GRAYSCALE);//以灰度方式加载图片

			//平滑中值滤波
			cvSmooth(src, src, CV_MEDIAN, 3, 0,0,0 );

			//形态学处理，膨胀与腐蚀
			if(src==NULL)return 0;
			//cvNamedWindow("Morph:Dilatee and Erod", 1);
			IplConvKernel*kernal=cvCreateStructuringElementEx(1,1,0,0,CV_SHAPE_RECT);
			IplConvKernel*kernal1=cvCreateStructuringElementEx(1,1,0,0,CV_SHAPE_RECT);
			IplConvKernel*kernal2=cvCreateStructuringElementEx(1,1,0,0,CV_SHAPE_RECT);
			cvDilate(src,src,kernal);
			cvErode(src,src,kernal2);
			cvDilate(src,src,kernal);
			cvErode(src,src,kernal2);
			cvReleaseStructuringElement(&kernal);
			cvReleaseStructuringElement(&kernal1);
			cvReleaseStructuringElement(&kernal2);

			IplImage* dst = cvCreateImage( cvGetSize(src), 8, 3 );

			//cvThreshold( src, src,51, 255, CV_THRESH_BINARY );//二值化
			cvAdaptiveThreshold( src,src, 255,0,1,91,5 ); //自适应二值化
			//cvNamedWindow( "Source", 1 );
			//cvShowImage( "Source", src );
			//提取轮廓
			cvFindContours( src, storage, &contour, sizeof(CvContour), 0, CV_CHAIN_APPROX_SIMPLE );
			cvZero( dst );//清空数组

			max_contour =GetAreaMaxContour(contour);//找出最大闭合面积的轮廓

			cvDrawContours( dst, max_contour, color, color, -1, 1, 8 );//绘制外部和内部的轮廓
			//	cvNamedWindow( "contour", 1 );
			//	cvShowImage( "contour", dst );
			//	Rect r0 = boundingRect(max_contour);
			//rectangle(result,r0,Scalar(0),2);
			//最小外接矩形
			CvBox2D rect = cvMinAreaRect2(max_contour);
			float height,width;
			height=rect.size.height;
			width=rect.size.width;
			float r=width/height;//宽长比
			//	W_L_Ratio[i]=r;
			trainingData[i][0]=r;
			//外接椭圆
			CvBox2D ellipse = cvFitEllipse2(max_contour);//最小二乘法的椭圆拟合
			float focal_len=abs(ellipse.center.x-ellipse.center.y);//焦距
			float e=focal_len/2/ellipse.size.width;//离心率
			//	eccentricity[i]=e;
			trainingData[i][1]=e;
			/*周长、面积
			float s,l;
			for(contour;contour!=NULL;contour=contour->h_next)
			{
			s=cvContourArea(contour,CV_WHOLE_SEQ);
			l=cvArcLength(contour,CV_WHOLE_SEQ,-1);//后面参数0表示轮廓不闭合，正数表示闭合；负数表示计算序列组成的面积；提取的角点以list形式时，用负数。

			}
			trainingData[i][5]=l/s;
			//trainingData[i][10]=s;*/

			//利用OpenCV函数求Hu矩
			CvMoments moments;
			CvHuMoments hu_moments;
			cvMoments(max_contour, &moments, 0); //调用 Opencv 的函数通过轮廓计算图像的空间矩和中心矩
			cvGetHuMoments(&moments, &hu_moments);//通过中心距计算图像的Hu矩存放到hu结构体中

			trainingData[i][2]=(float) hu_moments.hu1;
			trainingData[i][3]=(float)hu_moments.hu2;
			trainingData[i][4]=(float)hu_moments.hu3;
			//trainingData[i][5]=(float)hu_moments.hu4;
			//trainingData[i][6]=(float)hu_moments.hu5;
			//trainingData[i][7]=(float)hu_moments.hu6;
			//trainingData[i][8]=(float)hu_moments.hu7;
			return 0;
}
