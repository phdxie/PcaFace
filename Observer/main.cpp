#include<stdio.h>
#include<windows.h>
#include "PcaAlg.h"
const int IMAGE_WIDTH = 384;//图像列数
const int IMAGE_HEIGHT = 286;//图像行数
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
enum 
{
	EUCLID_DIST,
	SVM_ALG,
	BP_ANN,
};
typedef struct 
{
	int iClassLabel;
	char cFilePath[256];
}DataInfo;
DataInfo g_TrainDataInfo[]=
{
	{1,"..\\face\\Train\\1\\"},
	{2,"..\\face\\Train\\2\\"},
};
DataInfo g_TestDataInfo[]=
{
	{1,"..\\face\\Test\\1\\"},
	{2,"..\\face\\Test\\2\\"},
};
int GetAllFilesNum(char * pStr)
{
	int iNum=0;
	WIN32_FIND_DATA p;
	HANDLE h=FindFirstFile(strcat(pStr,"*.*"),&p);
	FindNextFile(h,&p);
	while(FindNextFile(h,&p))
	{
		iNum++;
	}
	return iNum;
}
void OutPutEigenVector(Mat EigenVector)
{
	for(int i=0;i<EigenVector.cols;i++)
	{
		int j=0;
		IplImage *pImage = cvCreateImage(cvSize(IMAGE_WIDTH,IMAGE_HEIGHT),IPL_DEPTH_32F,1);		
		for (size_t row=0; row < pImage->height; row++)
		{
			uchar* ptr = (uchar*)(pImage->imageData) + row * (pImage->widthStep);
			for (size_t cols=0; cols < pImage->width; cols++)
			{
				((float *)ptr)[cols]=EigenVector.at<float>(j,i);
				j++;
			}
		}
		char str[128]={0};
		sprintf(str,"..\\result\\EigenFace%d.jpg",i);
		cvSaveImage(str,pImage);
		cvReleaseImage(&pImage);		
	}
}

Mat LoadData(const char* fileName)
{
	IplImage *pImg=cvCreateImage(cvSize(IMAGE_WIDTH,IMAGE_HEIGHT),IPL_DEPTH_32F,1);
	pImg=cvLoadImage(fileName,0); 

	Mat RawData(IMAGE_SIZE,1,CV_32FC1);
	int j=0;
	for (size_t row=0;row < pImg->height; row++)
	{
		uchar* ptr = (uchar*)pImg->imageData + row * (pImg->width);
		for (size_t cols = 0; cols < pImg->width; cols++)
		{
			float fValue=ptr[cols];
			RawData.at<float>(j,0)=fValue;
			j++;
		}
	}
	cvReleaseImage(&pImg);
	return RawData;
}

void GetLabelData(Mat * pData, Mat * pDataLabel,
	   DataInfo * pDataInfo, int iDataInfoNum)
{
	int iDataNum=0;
	for(int i=0;i<iDataInfoNum;i++)
	{
		char str[256]={0};
		sprintf(str,"%s",pDataInfo[i].cFilePath);
		iDataNum=iDataNum+GetAllFilesNum(str);
	}
	Mat Data(IMAGE_SIZE,iDataNum,CV_32FC1);
	Mat DataLabel(1,iDataNum,CV_32SC1);
	int iEnd=0;
	for(int i=0;i<iDataInfoNum;i++)
	{
		char str[256]={0};
		sprintf(str,"%s",pDataInfo[i].cFilePath);
		int iBegin=iEnd;
		iEnd +=GetAllFilesNum(str);
		for(int j=iBegin;j<iEnd;j++)
		{
		    DataLabel.at<int>(0,j)=pDataInfo[i].iClassLabel;
		}
	}
	int iDataIdx=0;
	for(int iClassIdx=0;iClassIdx<iDataInfoNum;iClassIdx++)
	{
	    WIN32_FIND_DATA p;
		char str[256]={0};
		sprintf(str,"%s*.*",pDataInfo[iClassIdx].cFilePath);
		HANDLE h=FindFirstFile(str,&p);
		FindNextFile(h,&p);
     	while(FindNextFile(h,&p))
    	{
			string imageFileName= p.cFileName;
			char str[256]={0};
		    sprintf(str,"%s%s",pDataInfo[iClassIdx].cFilePath,imageFileName.data());
			Mat tmpData=LoadData( str);
			tmpData.col(0).copyTo(Data.col(iDataIdx));
			iDataIdx++;
		}
	}
	* pData=Data;
	* pDataLabel=DataLabel;
	return ;
}
void OutReconstructData(PCAAlg facePca,Mat TestData,int iClassIdx)
{
	facePca.ReconstructData(TestData);
	IplImage *pImage = cvCreateImage(cvSize(IMAGE_WIDTH,IMAGE_HEIGHT),IPL_DEPTH_32F,1); 	
	int j=0;
	for (size_t row=0; row < pImage->height; row++)
	{
		uchar* ptr = (uchar*)(pImage->imageData) + row * (pImage->widthStep);
		for (size_t cols=0; cols < pImage->width; cols++)
		{
			((float *)ptr)[cols]=TestData.at<float>(j,0);
			j++;
		}
	}
	char str1[128]={0};
	sprintf(str1,"..\\result\\ReconstructImg.jpg");
	cvSaveImage(str1,pImage);
	cvReleaseImage(&pImage);
}
#include <opencv2/ml/ml.hpp>
int PcaFeaClassify()
{
	PCAAlg facePca;
	Mat TrainData;
	Mat TrainDataLabel;
	GetLabelData(&TrainData,&TrainDataLabel,
		   g_TrainDataInfo, sizeof(g_TrainDataInfo)/sizeof(DataInfo));
	Mat TestData;
	Mat TestDataLabel;
	GetLabelData(&TestData,&TestDataLabel,
		   g_TestDataInfo, sizeof(g_TestDataInfo)/sizeof(DataInfo));

	facePca.InitTrainData(TrainData);	
	facePca.Trainer();
    Mat EigenFace=facePca.GetRawEigenVector();
	OutPutEigenVector(EigenFace);
	FILE * fp=fopen(LOG_FILE,"a+");
	fprintf(fp, "共获取到的特征向量个数为%d \n" ,EigenFace.cols);
	fclose(fp);

	int PCA_CLASSIFIER_TYPE=EUCLID_DIST;
	switch(PCA_CLASSIFIER_TYPE)
	{
	case EUCLID_DIST:
		{
			Mat PredictLabel=facePca.EuclidDistClassifier(TestData);
			FILE * fp=fopen(LOG_FILE,"a+");
			for(int idx=0;idx<TestData.cols;idx++)
			{
				int iPredictIdx=PredictLabel.at<int>(idx,0);
				fprintf(fp, "识别结果：测试数据%d 与训练数据%d最相似，属于类别%d \n" ,
						idx,iPredictIdx,TrainDataLabel.at<int>(0,iPredictIdx));
				//OutReconstructData(facePca,TestData,idx);
			}
			fclose(fp);
		}
		break;
	case SVM_ALG:
		break;
	case BP_ANN:
		break;
	default:
		break;
	}
	return 0;
}

int main()
{
	PcaFeaClassify();
	return 0;
}
