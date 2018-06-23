#include "dlt.h"

double radianToAngle(double radian) { return radian / CV_PI * 180; }

void DLT::Model::display(std::ostream &out)const
{
	double r3 = 1 / sqrt(m_l[8] * m_l[8] + m_l[9] * m_l[9] + m_l[10] * m_l[10]);
	double C = r3*r3*(m_l[0] * m_l[4] + m_l[1] * m_l[5] + m_l[2] * m_l[6]) - m_x0*m_y0;
	double A_ = r3*r3*(m_l[0] * m_l[0] + m_l[1] * m_l[1] + m_l[2] * m_l[2]) - m_x0*m_x0;
	double B = r3*r3*(m_l[4] * m_l[4] + m_l[5] * m_l[5] + m_l[6] * m_l[6]) - m_y0*m_y0;

	double fx = sqrt((A_*B - C*C) / B);
	double fy = sqrt((A_*B - C*C) / A_);

	double SINdBeta = C > 0 ? -sqrt(C*C / (A_*B)) : sqrt(C*C / (A_*B));
	double ds = -C / B / SINdBeta - 1;
	double dBeta = asin(SINdBeta);

	double a3 = r3*m_l[8];
	double b3 = r3*m_l[9];
	double c3 = r3*m_l[10];
	double b2 = (m_l[5] * r3 + b3*m_y0)*(1 + ds)*cos(dBeta) / fx;

	out << std::fixed << std::setprecision(6);
	out << "Li系数:  " << std::endl;
	for (auto i : m_l)
		out << i << "  ";
	out << std::endl << std::endl;

	out << "[fx,fy,x0,y0]" << std::endl;
	out << fx << "   " << fy << "   " << m_x0 << "   " << m_y0 << std::endl;
	out << std::endl;

	out.unsetf(std::ios::fixed);
	out << "[k1,k2,p1,p2]" << std::endl;
	out << m_k1 << "   " << m_k2 << "   " << m_p1 << "   " << m_p2 << std::endl;
	out << std::endl;
}

void DLT::Model::print()const
{
	display(std::cout);
}

void DLT::Model::storage(const std::string &file)const
{
	std::fstream fout(file, std::ios::out);
	if (!fout)
		return;

	display(fout);
}

DLT::Model DLT::sloveModel(const std::string &fileCPs, const std::string &fileIPs, bool storageRes, const std::string &file)const
{
	std::vector<Point2D> imgPoints;
	std::vector<Point3D> colPoints;
	if (!importControlPoints(fileCPs, colPoints) || !importImagePoints(fileIPs, imgPoints))
	{
		std::clog << "[error] file path error." << std::endl;

		system("pause");
		exit(0);
	}

	std::vector<Point3D> colPoints_;
	for (auto i = imgPoints.cbegin(); i != imgPoints.cend(); ++i)
		for (auto j = colPoints.cbegin(); j != colPoints.cend(); ++j)
		{
			if (i->idex == j->idex)
			{
				colPoints_.push_back(*j);
				break;
			}

			if (j == colPoints.cend() - 1)
			{
				std::clog << "[error] image points and control points do not match." << std::endl;
				std::clog << "image point " << i->idex << " can not find its matched control point." << std::endl;

				system("pause");
				exit(0);
			}
		}
	colPoints = colPoints_;
	colPoints_.clear();

	Model m;
	std::vector<Point2D> imgPoints_six;
	std::vector<Point3D> colPoints_six;

	std::default_random_engine e(2);
	std::uniform_int_distribution<int> u(0, imgPoints.size() - 1);
	for (int i = 0; i < 6; ++i)
	{
		size_t j = u(e);
		imgPoints_six.push_back(imgPoints[j]);
		colPoints_six.push_back(colPoints[j]);
	}

	approximateCalculateLi(imgPoints_six, colPoints_six, m);
	accurateCalculateLi(imgPoints, colPoints, m);

	if (storageRes)
	{
		cv::Mat_<double> A(cv::Size(15, imgPoints.size() * 2), 0);
		cv::Mat_<double> L(cv::Size(1, imgPoints.size() * 2), 0);

		for (size_t i = 0; i < A.rows; i += 2)
		{
			Point3D pt3D = colPoints[i / 2];
			Point2D pt2D = imgPoints[i / 2];

			double r2 = (pt2D.x - m.m_x0)*(pt2D.x - m.m_x0) + (pt2D.y - m.m_y0)*(pt2D.y - m.m_y0);
			double A_ = m.m_l[8] * pt3D.getX() + m.m_l[9] * pt3D.getY() + m.m_l[10] * pt3D.getZ() + 1;

			A.at<double>(i, 0) = -pt3D.getX() / A_;
			A.at<double>(i, 1) = -pt3D.getY() / A_;
			A.at<double>(i, 2) = -pt3D.getZ() / A_;
			A.at<double>(i, 3) = -1 / A_;
			A.at<double>(i, 4) = 0;
			A.at<double>(i, 5) = 0;
			A.at<double>(i, 6) = 0;
			A.at<double>(i, 7) = 0;
			A.at<double>(i, 8) = -pt2D.x*pt3D.getX() / A_;
			A.at<double>(i, 9) = -pt2D.x*pt3D.getY() / A_;
			A.at<double>(i, 10) = -pt2D.x*pt3D.getZ() / A_;
			A.at<double>(i, 11) = -(pt2D.x - m.m_x0)*r2;
			A.at<double>(i, 12) = -(pt2D.x - m.m_x0)*r2*r2;
			A.at<double>(i, 13) = -(r2 + 2 * (pt2D.x - m.m_x0)*(pt2D.x - m.m_x0));
			A.at<double>(i, 14) = -2 * (pt2D.x - m.m_x0)*(pt2D.y - m.m_y0);

			A.at<double>(i + 1, 0) = 0;
			A.at<double>(i + 1, 1) = 0;
			A.at<double>(i + 1, 2) = 0;
			A.at<double>(i + 1, 3) = 0;
			A.at<double>(i + 1, 4) = -pt3D.getX() / A_;
			A.at<double>(i + 1, 5) = -pt3D.getY() / A_;
			A.at<double>(i + 1, 6) = -pt3D.getZ() / A_;
			A.at<double>(i + 1, 7) = -1 / A_;
			A.at<double>(i + 1, 8) = -pt2D.y*pt3D.getX() / A_;
			A.at<double>(i + 1, 9) = -pt2D.y*pt3D.getY() / A_;
			A.at<double>(i + 1, 10) = -pt2D.y*pt3D.getZ() / A_;
			A.at<double>(i + 1, 11) = -(pt2D.y - m.m_y0)*r2;
			A.at<double>(i + 1, 12) = -(pt2D.y - m.m_y0)*r2*r2;
			A.at<double>(i + 1, 13) = -2 * (pt2D.x - m.m_x0)*(pt2D.y - m.m_y0);
			A.at<double>(i + 1, 14) = -(r2 + 2 * (pt2D.y - m.m_y0)*(pt2D.y - m.m_y0));

			L.at<double>(i, 0) = pt2D.x / A_;
			L.at<double>(i + 1, 0) = pt2D.y / A_;
		}

		cv::Mat_<double> Q = (A.t()*A).inv();
		cv::Mat_<double> X = Q*(A.t()*L);
		cv::Mat_<double> V = A*X - L;

		double vv = 0;
		std::vector<double> vi,mi;
		for (auto i = 0; i < V.rows; ++i)
		{
			double *data = V.ptr<double>(i);
			for (auto j = 0; j < V.cols; ++j)
			{
				vv += data[j] * data[j];
				vi.push_back(data[j]);
			}
		}

		double m0 = sqrt(vv / (2 * imgPoints.size() - 15));

		for (int i = 0; i < Q.rows; ++i)
		{
			double mi_ = sqrt(Q.at<double>(i, i))*m0;
			mi.push_back(mi_);
		}

		m.storage(file);

		std::fstream fout(file, std::ios::app);
		fout << std::fixed << std::setprecision(6);

		fout << "m0: " << m0 << std::endl;
		fout << std::endl;
	}

	return m;
}

void DLT::approximateCalculateLi(const std::vector<Point2D> &imgPts, const std::vector<Point3D> &colsPts, Model &m)const
{
	cv::Mat_<double> A(cv::Size(11, 11), 0);
	cv::Mat_<double> L(cv::Size(1, 11), 0);

	auto it_imgPts = imgPts.cbegin();
	auto it_colPts = colsPts.cbegin();
	for (size_t i = 0; i < 11; ++i)
	{
		if (i % 2 == 0)
		{
			A.at<double>(i, 0) = it_colPts->getX();
			A.at<double>(i, 1) = it_colPts->getY();
			A.at<double>(i, 2) = it_colPts->getZ();
			A.at<double>(i, 3) = 1;
			A.at<double>(i, 4) = 0;
			A.at<double>(i, 5) = 0;
			A.at<double>(i, 6) = 0;
			A.at<double>(i, 7) = 0;
			A.at<double>(i, 8) = it_imgPts->x*it_colPts->getX();
			A.at<double>(i, 9) = it_imgPts->x*it_colPts->getY();
			A.at<double>(i, 10) = it_imgPts->x*it_colPts->getZ();

			L.at<double>(i, 0) = -it_imgPts->x;
		}
		else
		{
			A.at<double>(i, 0) = 0;
			A.at<double>(i, 1) = 0;
			A.at<double>(i, 2) = 0;
			A.at<double>(i, 3) = 0;
			A.at<double>(i, 4) = it_colPts->getX();
			A.at<double>(i, 5) = it_colPts->getY();
			A.at<double>(i, 6) = it_colPts->getZ();
			A.at<double>(i, 7) = 1;
			A.at<double>(i, 8) = it_imgPts->y*it_colPts->getX();
			A.at<double>(i, 9) = it_imgPts->y*it_colPts->getY();
			A.at<double>(i, 10) = it_imgPts->y*it_colPts->getZ();

			L.at<double>(i, 0) = -it_imgPts->y;

			++it_imgPts;
			++it_colPts;
		}
	}

	cv::Mat_<double> Li = A.inv()*L;

	double x0 = 0, y0 = 0;
	x0 = -(Li.at<double>(0, 0)*Li.at<double>(8, 0) + Li.at<double>(1, 0)*Li.at<double>(9, 0) + Li.at<double>(2, 0)*Li.at<double>(10, 0))
		/ (Li.at<double>(8, 0)*Li.at<double>(8, 0) + Li.at<double>(9, 0)*Li.at<double>(9, 0) + Li.at<double>(10, 0)*Li.at<double>(10, 0));
	y0 = -(Li.at<double>(4, 0)*Li.at<double>(8, 0) + Li.at<double>(5, 0)*Li.at<double>(9, 0) + Li.at<double>(6, 0)*Li.at<double>(10, 0))
		/ (Li.at<double>(8, 0)*Li.at<double>(8, 0) + Li.at<double>(9, 0)*Li.at<double>(9, 0) + Li.at<double>(10, 0)*Li.at<double>(10, 0));

	m = Model(Li, 0, 0, 0, 0, x0, y0);
}

void DLT::accurateCalculateLi(const std::vector<Point2D> &imgPts, const std::vector<Point3D> &colPts, Model &m)const
{
	cv::Mat_<double> A(cv::Size(15, imgPts.size() * 2), 0);
	cv::Mat_<double> L(cv::Size(1, imgPts.size() * 2), 0);

	double x0_differ = 100, y0_differ = 100;
	size_t iterate_times = 0;
	while (x0_differ > 0.1 || y0_differ > 0.1)
	{
		for (size_t i = 0; i < A.rows; i += 2)
		{
			Point3D pt3D = colPts[i / 2];
			Point2D pt2D = imgPts[i / 2];

			double r2 = (pt2D.x - m.m_x0)*(pt2D.x - m.m_x0) + (pt2D.y - m.m_y0)*(pt2D.y - m.m_y0);
			double A_ = m.m_l[8] * pt3D.getX() + m.m_l[9] * pt3D.getY() + m.m_l[10] * pt3D.getZ() + 1;

			A.at<double>(i, 0) = -pt3D.getX() / A_;
			A.at<double>(i, 1) = -pt3D.getY() / A_;
			A.at<double>(i, 2) = -pt3D.getZ() / A_;
			A.at<double>(i, 3) = -1 / A_;
			A.at<double>(i, 4) = 0;
			A.at<double>(i, 5) = 0;
			A.at<double>(i, 6) = 0;
			A.at<double>(i, 7) = 0;
			A.at<double>(i, 8) = -pt2D.x*pt3D.getX() / A_;
			A.at<double>(i, 9) = -pt2D.x*pt3D.getY() / A_;
			A.at<double>(i, 10) = -pt2D.x*pt3D.getZ() / A_;
			A.at<double>(i, 11) = -(pt2D.x - m.m_x0)*r2;
			A.at<double>(i, 12) = -(pt2D.x - m.m_x0)*r2*r2;
			A.at<double>(i, 13) = -(r2 + 2 * (pt2D.x - m.m_x0)*(pt2D.x - m.m_x0));
			A.at<double>(i, 14) = -2 * (pt2D.x - m.m_x0)*(pt2D.y - m.m_y0);

			A.at<double>(i + 1, 0) = 0;
			A.at<double>(i + 1, 1) = 0;
			A.at<double>(i + 1, 2) = 0;
			A.at<double>(i + 1, 3) = 0;
			A.at<double>(i + 1, 4) = -pt3D.getX() / A_;
			A.at<double>(i + 1, 5) = -pt3D.getY() / A_;
			A.at<double>(i + 1, 6) = -pt3D.getZ() / A_;
			A.at<double>(i + 1, 7) = -1 / A_;
			A.at<double>(i + 1, 8) = -pt2D.y*pt3D.getX() / A_;
			A.at<double>(i + 1, 9) = -pt2D.y*pt3D.getY() / A_;
			A.at<double>(i + 1, 10) = -pt2D.y*pt3D.getZ() / A_;
			A.at<double>(i + 1, 11) = -(pt2D.y - m.m_y0)*r2;
			A.at<double>(i + 1, 12) = -(pt2D.y - m.m_y0)*r2*r2;
			A.at<double>(i + 1, 13) = -2 * (pt2D.x - m.m_x0)*(pt2D.y - m.m_y0);
			A.at<double>(i + 1, 14) = -(r2 + 2 * (pt2D.y - m.m_y0)*(pt2D.y - m.m_y0));

			L.at<double>(i, 0) = pt2D.x / A_;
			L.at<double>(i + 1, 0) = pt2D.y / A_;
		}

		cv::Mat_<double> X = (A.t()*A).inv()*(A.t()*L);

		double x0 = -(X.at<double>(0, 0)*X.at<double>(8, 0) + X.at<double>(1, 0)*X.at<double>(9, 0) + X.at<double>(2, 0)*X.at<double>(10, 0))
			/ (X.at<double>(8, 0)*X.at<double>(8, 0) + X.at<double>(9, 0)*X.at<double>(9, 0) + X.at<double>(10, 0)*X.at<double>(10, 0));
		double y0 = -(X.at<double>(4, 0)*X.at<double>(8, 0) + X.at<double>(5, 0)*X.at<double>(9, 0) + X.at<double>(6, 0)*X.at<double>(10, 0))
			/ (X.at<double>(8, 0)*X.at<double>(8, 0) + X.at<double>(9, 0)*X.at<double>(9, 0) + X.at<double>(10, 0)*X.at<double>(10, 0));

		x0_differ = fabs(x0 - m.m_x0), y0_differ = fabs(y0 - m.m_y0);

		cv::Mat_<double>Li(X, cv::Rect(cv::Point(0, 0), cv::Point(1, 11)));
		m = Model(Li, X.at<double>(11, 0), X.at<double>(12, 0), X.at<double>(13, 0), X.at<double>(14, 0), x0, y0);

		++iterate_times;
		if (iterate_times > 100)
		{
			std::clog << "收敛失败！" << std::endl;

			system("pause");
			exit(0);
		}
	}
	std::clog << "共迭代 " << iterate_times << " 次" << std::endl;
}

bool DLT::importImagePoints(const std::string &file, std::vector<Point2D> &imgPts)const
{
	std::fstream fin(file, std::ios::in);

	if (!fin)
		return false;

	size_t idex = 0;
	double x = 0, y = 0;
	while (!fin.eof())
	{
		fin >> idex;
		fin >> x;
		fin >> y;

		Point2D pt(idex, x, y);
		imgPts.push_back(pt);
	}
	fin.close();

	return true;
}

bool DLT::importControlPoints(const std::string &file, std::vector<Point3D> &colPts)const
{
	std::fstream fin(file, std::ios::in);

	if (!fin)
		return false;

	size_t idex = 0;
	double X = 0, Y = 0, Z = 0;
	while (!fin.eof())
	{
		fin >> idex;
		fin >> X;
		fin >> Y;
		fin >> Z;

		Point3D pt(idex, X, Y, Z);
		colPts.push_back(pt);
	}
	fin.close();

	return true;
}

void DLT::approximateCalculatePts(const std::vector<Point2D> &imgPots_l, const std::vector<Point2D> &imgPots_r, const Model &m_l, const Model &m_r, std::vector<Point3D> &pts)const
{
	for (size_t i = 0; i < imgPots_l.size();++i)
	{
		double xl = imgPots_l[i].x;
		double yl = imgPots_l[i].y;
		double xr = imgPots_r[i].x;
		double yr = imgPots_r[i].y;

		double r2_l = (xl - m_l.m_x0)*(xl - m_l.m_x0) + (yl - m_l.m_y0)*(yl - m_l.m_y0);
		double r2_r = (xr - m_r.m_x0)*(xr - m_r.m_x0) + (yr - m_r.m_y0)*(yr - m_r.m_y0);

		double x_l = xl + (xl - m_l.m_x0)*(r2_l*m_l.m_k1 + r2_l*r2_l*m_l.m_k2) + m_l.m_p1*(r2_l + 2 * (xl - m_l.m_x0)*(xl - m_l.m_x0)) + 2 * m_l.m_p2*(xl - m_l.m_x0)*(yl - m_l.m_y0);
		double y_l = yl + (yl - m_l.m_y0)*(r2_l*m_l.m_k1 + r2_l*r2_l*m_l.m_k2) + m_l.m_p2*(r2_l + 2 * (yl - m_l.m_y0)*(yl - m_l.m_y0)) + 2 * m_l.m_p1*(xl - m_l.m_x0)*(yl - m_l.m_y0);
		double x_r = xr + (xr - m_r.m_x0)*(r2_r*m_r.m_k1 + r2_r*r2_r*m_r.m_k2) + m_r.m_p1*(r2_r + 2 * (xr - m_r.m_x0)*(xr - m_r.m_x0)) + 2 * m_r.m_p2*(xr - m_r.m_x0)*(yr - m_r.m_y0);
	
		cv::Mat_<double>A(cv::Size(3, 3), 0);
		cv::Mat_<double>L(cv::Size(1, 3), 0);

		A.at<double>(0, 0) = m_l.m_l[0] + x_l*m_l.m_l[8];
		A.at<double>(0, 1) = m_l.m_l[1] + x_l*m_l.m_l[9];
		A.at<double>(0, 2) = m_l.m_l[2] + x_l*m_l.m_l[10];
		A.at<double>(2, 0) = m_r.m_l[0] + x_r*m_r.m_l[8];
		A.at<double>(2, 1) = m_r.m_l[1] + x_r*m_r.m_l[9];
		A.at<double>(2, 2) = m_r.m_l[2] + x_r*m_r.m_l[10];
		A.at<double>(1, 0) = m_l.m_l[4] + y_l*m_l.m_l[8];
		A.at<double>(1, 1) = m_l.m_l[5] + y_l*m_l.m_l[9];
		A.at<double>(1, 2) = m_l.m_l[6] + y_l*m_l.m_l[10];

		L.at<double>(0, 0) = -(m_l.m_l[3] + x_l);
		L.at<double>(1, 0) = -(m_l.m_l[7] + y_l);
		L.at<double>(2, 0) = -(m_r.m_l[3] + x_r);

		cv::Mat_<double>X = A.inv()*L;

		Point3D pt(imgPots_l[i].idex, X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0));
		pts.push_back(pt);
	}
}
void DLT::accurateCalculatePts(const std::vector<Point2D> &imgPots_l, const std::vector<Point2D> &imgPots_r, const Model &m_l, const Model &m_r, std::vector<Point3D> &pts)const
{
	cv::Mat_<double>A(cv::Size(3, 4), 0);
	cv::Mat_<double>L(cv::Size(1, 4), 0);

	double X_differ = 10000, Y_differ = 10000, Z_differ = 10000;
	for (size_t i = 0; i < pts.size(); ++i)
	{
		double xl = imgPots_l[i].x;
		double yl = imgPots_l[i].y;
		double xr = imgPots_r[i].x;
		double yr = imgPots_r[i].y;

		double r2_l = (xl - m_l.m_x0)*(xl - m_l.m_x0) + (yl - m_l.m_y0)*(yl - m_l.m_y0);
		double r2_r = (xr - m_r.m_x0)*(xr - m_r.m_x0) + (yr - m_r.m_y0)*(yr - m_r.m_y0);

		double x_l = xl + (xl - m_l.m_x0)*(r2_l*m_l.m_k1 + r2_l*r2_l*m_l.m_k2) + m_l.m_p1*(r2_l + 2 * (xl - m_l.m_x0)*(xl - m_l.m_x0)) + 2 * m_l.m_p2*(xl - m_l.m_x0)*(yl - m_l.m_y0);
		double y_l = yl + (yl - m_l.m_y0)*(r2_l*m_l.m_k1 + r2_l*r2_l*m_l.m_k2) + m_l.m_p2*(r2_l + 2 * (yl - m_l.m_y0)*(yl - m_l.m_y0)) + 2 * m_l.m_p1*(xl - m_l.m_x0)*(yl - m_l.m_y0);
		double x_r = xr + (xr - m_r.m_x0)*(r2_r*m_r.m_k1 + r2_r*r2_r*m_r.m_k2) + m_r.m_p1*(r2_r + 2 * (xr - m_r.m_x0)*(xr - m_r.m_x0)) + 2 * m_r.m_p2*(xr - m_r.m_x0)*(yr - m_r.m_y0);
		double y_r = yr + (yr - m_r.m_y0)*(r2_r*m_r.m_k1 + r2_r*r2_r*m_r.m_k2) + m_r.m_p2*(r2_r + 2 * (yr - m_r.m_y0)*(yr - m_r.m_y0)) + 2 * m_r.m_p1*(xr - m_r.m_x0)*(yr - m_r.m_y0);

		while (X_differ > 0.1 || Y_differ > 0.1 || Z_differ > 0.1)
		{
			double A_l = m_l.m_l[8] * pts[i].getX() + m_l.m_l[9] * pts[i].getY() + m_l.m_l[10] * pts[i].getZ() + 1;
			double A_r = m_r.m_l[8] * pts[i].getX() + m_r.m_l[9] * pts[i].getY() + m_r.m_l[10] * pts[i].getZ() + 1;

			A.at<double>(0, 0) = -(m_l.m_l[0] + x_l*m_l.m_l[8]) / A_l;
			A.at<double>(0, 1) = -(m_l.m_l[1] + x_l*m_l.m_l[9]) / A_l;
			A.at<double>(0, 2) = -(m_l.m_l[2] + x_l*m_l.m_l[10]) / A_l;
			A.at<double>(2, 0) = -(m_r.m_l[0] + x_r*m_r.m_l[8]) / A_r;
			A.at<double>(2, 1) = -(m_r.m_l[1] + x_r*m_r.m_l[9]) / A_r;
			A.at<double>(2, 2) = -(m_r.m_l[2] + x_r*m_r.m_l[10]) / A_r;
			A.at<double>(1, 0) = -(m_l.m_l[4] + y_l*m_l.m_l[8]) / A_l;
			A.at<double>(1, 1) = -(m_l.m_l[5] + y_l*m_l.m_l[9]) / A_l;
			A.at<double>(1, 2) = -(m_l.m_l[6] + y_l*m_l.m_l[10]) / A_l;
			A.at<double>(3, 0) = -(m_r.m_l[4] + y_r*m_r.m_l[8]) / A_r;
			A.at<double>(3, 1) = -(m_r.m_l[5] + y_r*m_r.m_l[9]) / A_r;
			A.at<double>(3, 2) = -(m_r.m_l[6] + y_r*m_r.m_l[10]) / A_r;

			L.at<double>(0, 0) = (m_l.m_l[3] + x_l) / A_l;
			L.at<double>(1, 0) = (m_l.m_l[7] + y_l) / A_l;
			L.at<double>(2, 0) = (m_r.m_l[3] + x_r) / A_r;
			L.at<double>(3, 0) = (m_r.m_l[7] + y_r) / A_r;

			cv::Mat_<double>X = (A.t()*A).inv()*(A.t()*L);

			X_differ = fabs(pts[i].getX() - X.at<double>(0, 0));
			Y_differ = fabs(pts[i].getY() - X.at<double>(1, 0));
			Z_differ = fabs(pts[i].getZ() - X.at<double>(2, 0));

			pts[i] = Point3D(pts[i].idex, X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0));
		}
	}
}

std::shared_ptr<std::vector<DLT::Point3D>> DLT::compute(const std::string &imgPointsfile_l, const std::string &imgPointsfile_r, const Model &m_l, const Model &m_r)const
{
	std::vector<Point2D> pts_l, pts_r;

	if (!importImagePoints(imgPointsfile_l, pts_l) || !importImagePoints(imgPointsfile_r, pts_r))
	{
		std::clog << "[error] file path error" << std::endl;

		system("pause");
		exit(0);
	}

	std::vector<Point2D> pts_l_,pts_r_;
	for (auto i = pts_l.cbegin(); i != pts_l.cend(); ++i)
		for (auto j = pts_r.cbegin(); j != pts_r.cend(); ++j)
		{
			if (i->idex == j->idex)
			{
				pts_l_.push_back(*i);
				pts_r_.push_back(*j);
				break;
			}
		}

	pts_l = pts_l_;
	pts_r = pts_r_;
	pts_l_.clear();
	pts_r_.clear();

	std::vector<Point3D> pts;

	approximateCalculatePts(pts_l, pts_r, m_l, m_r, pts);
	accurateCalculatePts(pts_l, pts_r, m_l, m_r, pts);

	return std::make_shared<std::vector<DLT::Point3D>>(pts);
}