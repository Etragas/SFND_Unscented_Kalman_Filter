#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd PredictSigma(MatrixXd& Xsig_aug, float delta_t);
void AugmentedSigmaPoints(UKF& self, MatrixXd* Xsig_out);
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  is_initialized_ = false;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  x_.setZero();
  P_.setIdentity();
  P_(0, 0) = std_laspx_ * std_laspx_;
  P_(1, 1) = std_laspy_ * std_laspy_;
  P_(2, 2) = 1;
  P_(3, 3) = std_radphi_ * std_radphi_;
  P_(4, 4) = std_radrd_ * std_radrd_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  std::cout << meas_package.timestamp_ << std::endl;
  if (!is_initialized_) {
    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
      time_us_ = meas_package.timestamp_;
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      x_(2) = 0;
      is_initialized_ = true;
    }
    else {
      time_us_ = meas_package.timestamp_;
      return;
    }
  }

  double delta_t = (double)meas_package.timestamp_ - (double)time_us_;
  delta_t /= 1000000;
  std::cout << delta_t << std::endl;
  Prediction(delta_t);
  time_us_ = meas_package.timestamp_;
  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
    UpdateLidar(meas_package);
  }
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // if (delta_t == 0) return;
  MatrixXd* sigmaPoints = new MatrixXd();
  AugmentedSigmaPoints(*this, sigmaPoints);
  Xsig_pred_ = PredictSigma(*sigmaPoints, delta_t);
    // set weights
  int n_aug = 7;
  double lambda = 3 - n_aug;
  VectorXd weights(2*n_aug+1);
  weights(0) = lambda / (lambda + n_aug);
  for (int i =1; i < 2*n_aug+1; i++) {
      weights(i) = 1 / (2 * (lambda + n_aug));
  }
  
  // predict state mean
  x_ = (Xsig_pred_  * weights).rowwise().sum();
  
  MatrixXd diff = Xsig_pred_.colwise() - x_;
  P_ = diff * weights.asDiagonal();
  P_ *= diff.transpose();
  
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  
  // Take sigma points, move them forward by dt using process
  
  //std::cout << "Start: " << std::endl  << std::endl;
  int n_aug = 7;
  double lambda = 3 - n_aug;
  MatrixXd Zsig(2, 2 * n_aug + 1);
  VectorXd weights(2*n_aug+1);
  weights(0) = lambda / (lambda + n_aug);
  for (int i =1; i < 2*n_aug+1; i++) {
      weights(i) = 1 / (2 * (lambda + n_aug));
  }

  //std::cout << "Weights" << std::endl << weights << std::endl;
  for (int i = 0; i < 2 * n_aug + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }
  //std::cout << "ZS: " << std::endl << Zsig << std::endl;

  VectorXd z_pred(2);
  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; ++i) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }
  // z_pred(2) += 0.00001;
  //std::cout << "ZP: " << std::endl << z_pred << std::endl;



  MatrixXd S = MatrixXd(2,2);
  S.fill(0.0);
  MatrixXd z_diff(2, 2*n_aug+1);
  for (int i = 0; i < 2 * n_aug + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff_col = Zsig.col(i) - z_pred;

    S = S + weights(i) * z_diff_col * z_diff_col.transpose();
  }
  //std::cout << "S: " << std::endl << S << std::endl;

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(2, 2);
  R <<  std_laspx_*std_laspx_, 0,
        0, std_radphi_*std_radphi_;
  S = S + R;
  
  MatrixXd state_diff = Xsig_pred_.colwise() - x_;
  MatrixXd measurement_diff = Zsig.colwise() - z_pred;
  MatrixXd Tc = state_diff * weights.asDiagonal() * measurement_diff.transpose();
  //std::cout << "TC: " << std::endl << Tc << std::endl;
  MatrixXd K = Tc*S.inverse();
  //std::cout << "K: " << std::endl << K << std::endl;

  x_ = x_ + K*(meas_package.raw_measurements_ -z_pred);
  //std::cout << "X: " << std::endl << x_ << std::endl;
  P_ = P_ - (K * S * K.transpose());
  //std::cout << "P: " << std::endl << P_ << std::endl;
  // calculate cross correlation matrix


}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  //std::cout << "Start: " << std::endl  << std::endl;
  int n_aug = 7;
  double lambda = 3 - n_aug;
  MatrixXd Zsig(3, 2 * n_aug + 1);
  VectorXd weights(2*n_aug+1);
  weights(0) = lambda / (lambda + n_aug);
  for (int i =1; i < 2*n_aug+1; i++) {
      weights(i) = 1 / (2 * (lambda + n_aug));
  }
  //std::cout << "Weights" << std::endl << weights << std::endl;
  for (int i = 0; i < 2 * n_aug + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }
  //std::cout << "ZS: " << std::endl << Zsig << std::endl;

  VectorXd z_pred(3);
  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; ++i) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }
  // z_pred(2) += 0.00001;
  //std::cout << "ZP: " << std::endl << z_pred << std::endl;



  MatrixXd S = MatrixXd(3,3);
  S.fill(0.0);
  MatrixXd z_diff(3, 2*n_aug+1);
  for (int i = 0; i < 2 * n_aug + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff_col = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff_col(1)> M_PI) z_diff_col(1)-=2.*M_PI;
    while (z_diff_col(1)<-M_PI) z_diff_col(1)+=2.*M_PI;

    S = S + weights(i) * z_diff_col * z_diff_col.transpose();
  }
  //std::cout << "S: " << std::endl << S << std::endl;

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(3, 3);
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
  S = S + R;
  
  // MatrixXd state_diff = Xsig_pred_.colwise() - x_;
  // MatrixXd measurement_diff = Zsig.colwise() - z_pred;
  // MatrixXd Tc = state_diff * weights.asDiagonal() * measurement_diff.transpose();
  MatrixXd Tc = MatrixXd(5, 3);

  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff_col = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff_col(1)> M_PI) z_diff_col(1)-=2.*M_PI;
    while (z_diff_col(1)<-M_PI) z_diff_col(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights(i) * x_diff * z_diff_col.transpose();
  }

  //std::cout << "TC: " << std::endl << Tc << std::endl;
  MatrixXd K = Tc*S.inverse();
  //std::cout << "K: " << std::endl << K << std::endl;

  x_ = x_ + 0.3*K*(meas_package.raw_measurements_ -z_pred);
  //std::cout << "X: " << std::endl << x_ << std::endl;
  P_ = P_ - 0.3*(K * S * K.transpose());
  //std::cout << "P: " << std::endl << P_ << std::endl;
  // calculate cross correlation matrix

}

void AugmentedSigmaPoints(UKF& self, MatrixXd* Xsig_out) {

  // set state dimension
  int n_x = 5;

  // set augmented dimension
  int n_aug = 7;

  // define spreading parameter
  double lambda = 3 - n_aug;

  // set example state

  // create augmented mean vector
  int nx = n_aug;

  VectorXd x_aug = VectorXd(7);
  x_aug.head(5) = self.x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7) =  MatrixXd::Zero(7, 7);

  P_aug.topLeftCorner(5,5) = self.P_;
  P_aug(5,5) = self.std_a_ * self.std_a_;
  P_aug(6,6) = self.std_yawdd_ * self.std_yawdd_;
  MatrixXd A = P_aug.llt().matrixL();
  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  /**
   * Student part begin
   */

  // create augmented mean state
   Xsig_aug.col(0) = x_aug;
  for (int i = 1; i < nx+1; i++){
    Xsig_aug.col(i) = (x_aug + sqrt(lambda + nx)*A.col(i-1));
  }

  for (int i = nx+1; i < 2*nx+1; i++){
    Xsig_aug.col(i) = (x_aug - sqrt(lambda + nx)*A.col(i-nx-1));
  }

  // create augmented covariance matrix

  // create square root matrix

  // create augmented sigma points
  
  /**
   * Student part end
   */

  // print result
  //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  // write result
  *Xsig_out = Xsig_aug;
}

MatrixXd PredictSigma(MatrixXd& Xsig_aug, float delta_t) {
  // predict sigma points
  int n_aug = 7;
  MatrixXd Xsig_pred = MatrixXd(5, 2 * n_aug + 1);
  for (int i = 0; i< 2*n_aug+1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    (Xsig_pred)(0,i) = px_p;
    (Xsig_pred)(1,i) = py_p;
    (Xsig_pred)(2,i) = v_p;
    (Xsig_pred)(3,i) = yaw_p;
    (Xsig_pred)(4,i) = yawd_p;
  }
  return Xsig_pred;
}