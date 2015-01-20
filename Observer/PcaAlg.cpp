#include "PcaAlg.h"

#define EIGEN_THRESHOLD 0//295872960.000000  
void OutPutAllDataFeas(Mat DataFeas)
{
	FILE * fp=fopen(LOG_FILE,"a+");
	fprintf(fp,"\nData fea is: \n");
	for(int i=0;i<DataFeas.cols;i++)
	{
		fprintf(fp,"Data reduced fea :");
		for(int j=0;j<DataFeas.rows;j++)
		{
			fprintf(fp," %.4f ",DataFeas.at<float>(j,i));
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}
void DimReduce(Mat tmpEigenValue,Mat tmpEigenVector,
	Mat * pEigenVectorDimReduced,Mat * pEigenValueDimReduced, int * pEigenNum)
{
	* pEigenNum=0;
	for(int i = 0;i < tmpEigenValue.cols; i++)
	{
		float tmp=tmpEigenValue.at <float>(i,i);
		if( tmp > EIGEN_THRESHOLD)   
		{    
		    (* pEigenNum)++; 
		}
		FILE * fp=fopen(LOG_FILE,"a+");
		fprintf(fp, "特征向量 %d 特征值为 %f \n" ,i,tmp);
		fclose(fp);
	}
	assert(* pEigenNum!=0);
	Mat EigenVectorDimReduced( tmpEigenValue.cols,* pEigenNum,CV_32FC1 );
	Mat EigenValueDimReduced( * pEigenNum,1,CV_32FC1 );
	int k=0;
	for(int i = 0;i < tmpEigenVector.cols; i++)
	{
		if(tmpEigenValue.at <float>(i,i) > EIGEN_THRESHOLD) 
		{
			tmpEigenVector.col(i).copyTo(EigenVectorDimReduced.col(k));
			EigenValueDimReduced.at <float>(k,0)= tmpEigenValue.at <float>(i,i);
			k++;
		}
	}
	 * pEigenVectorDimReduced=EigenVectorDimReduced;
	 * pEigenValueDimReduced=EigenValueDimReduced;
}
Mat PCAAlg::NormalizeTestData(Mat FeaData)
{			
	int iFeaNum=GetEigenVector().cols;
	Mat NormalizedFeas(iFeaNum,FeaData.cols,CV_32FC1);
	for(int i=0;i<FeaData.cols;i++)
	{
		Mat FeaDataMean = FeaData.col(i) - GetDataMean();
		Mat FeaDataFeas = GetEigenVector().t() * FeaDataMean;
			
		for(int j=0;j<iFeaNum;j++)
		{
			NormalizedFeas.at<float>(j,i)=(FeaDataFeas.at<float>(j,0)-m_fMin.at<float>(j,0))
				/(m_fMax.at<float>(j,0)-m_fMin.at<float>(j,0));
		}
	}
	return NormalizedFeas;
}
Mat PCAAlg::EuclidDistClassifier(Mat TestData)
{
	int iTestNum=TestData.cols;
	Mat ClassifyResult(iTestNum,1,CV_32SC1);
	Mat TrainDataFeas = GetTrainDataDimReduced();

	Mat TestDataFeas=NormalizeTestData(TestData);	
	OutPutAllDataFeas(TestDataFeas);
	int iTrainNum = GetTrainNum();
	for(int iTestDataIdx=0;iTestDataIdx<iTestNum;iTestDataIdx++)
	{		
		
		Mat EuclidDist(1, iTrainNum,CV_32FC1);
		for(int i = 0; i < iTrainNum; i++)
		{
			EuclidDist.at<float>(0,i) = cvNorm(&CvMat(TestDataFeas.col(iTestDataIdx)),
								   &CvMat(TrainDataFeas.col(i)),CV_L2);
		}
		double minDist = EuclidDist.at<float>(0,0);
		int minIdx = 0;
		for(int i = 0; i < iTrainNum; i++)
		{
			if( EuclidDist.at<float>(0,i) < minDist)
			{
				minDist = EuclidDist.at<float>(0,i);
				minIdx = i;
			}
		}
		ClassifyResult.at<int>(iTestDataIdx,0)=minIdx;
	}
	return ClassifyResult;
}


void PCAAlg::InitTrainData( Mat TrainData)
{
	m_TrainData=TrainData;
	m_iTrainNum=TrainData.cols;
}
void PCAAlg::CalDataMean(Mat TrainData)  
{
	Mat dataMean(TrainData.rows, 1,CV_32FC1);

	for(int i = 0; i < TrainData.rows; i++)
	{
		float sum = 0;
		for(int j = 0;j < TrainData.cols;j++)
		{
			sum += TrainData.at<float>(i,j);
		}
		sum = sum /TrainData.cols; 
		dataMean.at<float>(i,0) = sum;
	}
	m_MeanData = dataMean;
}
void PCAAlg::CalEigenVector()
{
	int iRawDataLen=m_TrainData.rows;
	Mat NormalData(iRawDataLen, m_iTrainNum,CV_32FC1);
	for(int i=0;i < iRawDataLen;i++)
	{
		for(int j=0;j < m_iTrainNum;j++)
		{
		    //求矩阵Data跟矩阵Data_mean的差值
			NormalData.at<float>(i,j)= m_TrainData.at<float>(i,j) - m_MeanData.at<float>(i,0);
		}
	}

	Mat CovMatrix(m_iTrainNum,m_iTrainNum,CV_32FC1);
	//求协方差矩阵
	CovMatrix = NormalData.t()*NormalData;

	Mat tmpEigenValue(m_iTrainNum,m_iTrainNum,CV_32FC1);
	Mat tmpEigenVector(m_iTrainNum,m_iTrainNum,CV_32FC1);
	//tmpEigenVector 为协方差矩阵的特征向量矩阵，
	//tmpEigenValue 为协方差矩阵的特征值对角矩阵

	cvSVD(&CvMat(CovMatrix), &CvMat(tmpEigenValue), NULL, &CvMat(tmpEigenVector));
	
	Mat EigenVectorDimReduced;
	Mat EigenValueDimReduced;
	int iEigenNum=0;
    DimReduce(tmpEigenValue,tmpEigenVector,&EigenVectorDimReduced,&EigenValueDimReduced,&iEigenNum);

	Mat EigenVector(iRawDataLen,EigenVectorDimReduced.cols,CV_32FC1);
	EigenVector = NormalData * EigenVectorDimReduced;//得到特征矩阵
	m_RawEigenVector=EigenVector.clone();
	//特征矩阵归一化
	for(int i=0;i<EigenVectorDimReduced.cols;i++)
	{
		for(int j=0;j<iRawDataLen;j++)
		{
			EigenVector.at<float>(j,i) = EigenVector.at<float>(j,i) / sqrt(EigenValueDimReduced.at<float>(i,0));
		}
	}

	m_NormalEigenVector = EigenVector;

	//向新的特征空间进行投影，即对原始数据进行降维
    Mat DataDimReduced(m_NormalEigenVector.cols,m_iTrainNum,CV_32FC1);
    DataDimReduced = m_NormalEigenVector.t()* NormalData;
    m_DataDimReduced = DataDimReduced;
	OutPutAllDataFeas(m_DataDimReduced);
	//下面进行归一化，但是归一化和不归一化的结果是一样的？？
	Mat fMax(m_DataDimReduced.rows,1,CV_32FC1);
	Mat fMin(m_DataDimReduced.rows,1,CV_32FC1);
    for(int i=0;i<m_DataDimReduced.rows;i++)
	{
		fMax.at<float>(i,0)=-10000000;
		fMin.at<float>(i,0)=10000000;
		for(int j=0;j<m_DataDimReduced.cols;j++)
		{
			if(fMax.at<float>(i,0)<m_DataDimReduced.at<float>(i,j))
			{
				fMax.at<float>(i,0)=m_DataDimReduced.at<float>(i,j);
			}
			if(fMin.at<float>(i,0)>m_DataDimReduced.at<float>(i,j))
			{
			    fMin.at<float>(i,0)=m_DataDimReduced.at<float>(i,j);
			}
		}
	}
	m_fMin=fMin;
	m_fMax=fMax;
	for(int i=0;i<m_DataDimReduced.cols;i++)
	{
		for(int j=0;j<m_DataDimReduced.rows;j++)
		{
			m_DataDimReduced.at<float>(j,i)=(m_DataDimReduced.at<float>(j,i)-m_fMin.at<float>(j,0))/
				(m_fMax.at<float>(j,0)-m_fMin.at<float>(j,0));
		}
	}
	OutPutAllDataFeas(m_DataDimReduced);
}
Mat PCAAlg::GetTrainDataDimReduced()
{
	return m_DataDimReduced;
}
Mat PCAAlg::ReconstructData(Mat Data)
{
	Mat DataMean = Data - GetDataMean();
	Mat DataFeas = GetEigenVector().t() * DataMean;
	return m_NormalEigenVector*DataFeas+m_MeanData;
}

void PCAAlg::Trainer()
{
	CalDataMean(m_TrainData);
	CalEigenVector();
}
