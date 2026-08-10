#include <cmath>
#include <utility>
#include <limits>
#include <array>
#include <mdspan>
#include <algorithm>
#include <random>
#include <filesystem>
#include <print>

#include <hdf5.h>
#include <hdf5_hl.h>

const double FP_TOLERANCE=1e-14;
// ========================================================================= //
// RNG
// ========================================================================= //
std::mt19937_64 RNG{std::random_device{}()};
std::normal_distribution<double> NORMAL_DISTRIBUTION(0.0, 1.0);
std::uniform_real_distribution<double> UNIFORM_DISTRIBUTION(0.0, 1.0);

// ========================================================================= //
// Particle declaration
// ========================================================================= //
struct Cell;
template <size_t XBins, size_t MuBins> class HistogramTally;

struct Particle {

  double x;
  double mu;  
  bool alive;

  double distance_to_next_collision;
  
  Cell* cell;

  Particle(const double x, const double mu, Cell* cell)
    : x(x),
      mu(mu),
      alive(true),
      distance_to_next_collision(PathLengthToNextCollision()),
      cell(cell) {}

  void DoHardCollision();

  template <size_t XBins, size_t MuBins> 
  void Update(double ds, HistogramTally<XBins, MuBins>& tally);

  double PathLengthToNextCollision() {return -std::log(UNIFORM_DISTRIBUTION(RNG));} 
};

// ========================================================================= //
// Geometry stuff
// ========================================================================= //
struct Material {
  double sigma_tr;
  double sigma_t;
};

struct Cell {
  const double xmin;
  const double xmax;
  const Material& mat;

  Cell* right = nullptr;
  Cell* left = nullptr;

  bool Contains(const Particle& p) const {
    return p.x >= xmin && p.x <= xmax;
  }

  double Distance(const Particle& p) const {
    if (p.mu > 0.0) {
      return xmax - p.x;
    } else if (p.mu < 0.0) {
      return p.x - xmin;
    } else {
      return std::numeric_limits<double>::infinity();
    }
  }

  std::pair<Cell*, double> ShiftCells(Particle& p, double ds) const {
    if (p.mu > 0.0) {
      double diff_ds = (xmax - p.x) / p.mu;
      p.x = xmax;
      return {right, diff_ds};
    } else if (p.mu < 0.0) { 
      double diff_ds = (xmin - p.x) / p.mu;
      p.x = xmin;
      return {left, diff_ds};
    } else {
      p.alive = false;
      return {nullptr, 0.0};
    }
  }
};

// ========================================================================= //
// Mesh stuff
// ========================================================================= //
struct Mesh {
  const double _min;
  const double _max;
  const size_t _n_bins;

  size_t GetIndex(double value) const {
    auto rel_position = static_cast<double>(_n_bins) * ((value - _min) / (_max - _min));
    auto floor = std::floor(rel_position);
    return std::clamp<size_t>(static_cast<size_t>(floor), 0, _n_bins - 1);
  }

  double Width() const {
    return (_max - _min) / _n_bins;
  }
  double BinCenter(size_t bin) const {
    double width = Width();
    return bin*width + width/2;
  }

  std::pair<double, double> BinBounds(size_t bin) const {
    auto lower = _min + bin * Width();
    auto upper = lower + Width();
    return {lower, upper};
  }
};

// ========================================================================= //
// Physics
// ========================================================================= //

namespace EulerMaruyama {
  static double NewX(const Particle& p, double ds) {
    return p.x + p.mu * ds;
  }
  static double NewMu(const Particle& p, double ds) {
    auto trxs = p.cell->mat.sigma_tr;
    auto mn = p.mu;
    double xi = NORMAL_DISTRIBUTION(RNG);
    mn -= trxs * mn * ds + std::sqrt(trxs * (1 - mn * mn) * ds) * xi;
    return std::clamp(mn, -1.0, 1.0);
  }
}

// ========================================================================= //
// Tally stuff
// ========================================================================= //
template <size_t XBins, size_t MuBins>
class HistogramTally {
  public:
    HistogramTally(const Mesh& x, const Mesh& mu): x_mesh(x), mu_mesh(mu){
      namespace fs = std::filesystem;
      fs::path fpath = "sde_output.h5";
      if (fs::exists(fpath)) fs::remove(fpath); 
      
      file_id = H5Fcreate(fpath.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    }

    ~HistogramTally() {
      H5Fclose(file_id);
    }

    // x then mu
    std::array<double, XBins * MuBins> _tally{};
    // mesh
    const Mesh& x_mesh;
    const Mesh& mu_mesh;

    void Score(double x, double mu, double ds) {
      if (ds <= 0.0) return;

      auto mu_index = mu_mesh.GetIndex(mu);
      double dx = x_mesh.Width();
      
      if (std::abs(mu) < FP_TOLERANCE) {
        auto x_index = x_mesh.GetIndex(x);
        _tally[x_index * MuBins + mu_index] += ds;
        return;
      }

      do {
        auto x_index = x_mesh.GetIndex(x);
        auto [lower, upper] = x_mesh.BinBounds(x_index); 
        double distance_to_edge = mu > 0.0 ? upper - x : x - lower;

        bool at_edge = (mu > 0.0 && x_index == XBins - 1) || (mu < 0.0 && x_index ==0);
        if (distance_to_edge <= FP_TOLERANCE) {
          if (at_edge) {
            _tally[x_index * MuBins + mu_index] += ds;
            return;
          }
          x += (mu > 0.0 ? 1.0 : -1.0) * 2 * FP_TOLERANCE;
          continue;
        }

        double path_to_edge = distance_to_edge / std::abs(mu);
        double local_ds = std::min(path_to_edge, ds);

        _tally[x_index * MuBins + mu_index] += local_ds;
        x += mu * local_ds;
        ds -= local_ds;

      } while (ds > FP_TOLERANCE);
    }

    void Reset() {
      _tally.fill(0.0);
    }

    void WriteOut(std::string dataset, size_t num_histories) {
      auto group = H5Gcreate(file_id, dataset.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      std::array<double, XBins> x_centers;
      std::array<double, XBins> phi{};
      for (size_t x = 0; x < XBins; x++) {
        x_centers[x] = x_mesh.BinCenter(x); 
        for (size_t m = 0; m < MuBins; m++)
        {
          _tally[x * MuBins + m] /= static_cast<double>(num_histories) * mu_mesh.Width() * x_mesh.Width();
          phi[x] += _tally[x * MuBins + m];
        }
        phi[x] *= mu_mesh.Width() ;
      }

      hsize_t xdims[1] = {XBins};
      hsize_t bothdims[2] = {XBins, MuBins};
      H5LTmake_dataset_double(group, "x_centers", 1, xdims, x_centers.data());
      H5LTmake_dataset_double(group, "phi", 1, xdims, phi.data());
      H5LTmake_dataset_double(group, "psi", 2, bothdims, _tally.data());

      H5Gclose(group);
    }
  
  private:
    hid_t file_id;
};

// ========================================================================= //
// Particle instantation
// ========================================================================= //
void Particle::DoHardCollision() {
  distance_to_next_collision = PathLengthToNextCollision();
}

template <size_t XBins, size_t MuBins>
void Particle::Update(double ds, HistogramTally<XBins, MuBins>& tally) {
  do {
    double x0 = x;
    double mu0 = mu;
    double distance_to_surf = cell->Distance(*this);
    double potential_x = EulerMaruyama::NewX(*this, ds);
    double ds_to_coll = distance_to_next_collision / cell->mat.sigma_t;
    bool leak = (std::abs(potential_x - x) >= distance_to_surf);
    bool collide_before_leak = (distance_to_surf >= std::abs(ds_to_coll * mu));
    bool collide = (ds >= ds_to_coll) && collide_before_leak;

    if (leak && !collide_before_leak) {
      auto [new_cell, diff_ds] = cell->ShiftCells(*this, ds);
      tally.Score(x0, mu0, diff_ds);
      mu = EulerMaruyama::NewMu(*this, diff_ds);
      distance_to_next_collision -= (alive) ? diff_ds * cell->mat.sigma_t : 0.0;
      cell = new_cell;
      ds -= diff_ds;
    } else if (collide) {
      tally.Score(x0, mu0, ds_to_coll);
      x = EulerMaruyama::NewX(*this, ds_to_coll);
      DoHardCollision();
      mu = EulerMaruyama::NewMu(*this, ds_to_coll);
      ds -= ds_to_coll;
    } else {
      tally.Score(x0, mu0, ds);
      x = potential_x;
      mu = EulerMaruyama::NewMu(*this, ds);
      distance_to_next_collision -= ds * cell->mat.sigma_t;
      ds = 0.0;
    }
    // verify in cell
    // first check is nullptr check to make sure this->cell isnt a null
    alive = cell ? cell->Contains(*this) : false;
  } while (ds > 0.0 && alive);
}

// ========================================================================= //
// Solver stuff
// ========================================================================= //

int main() {

  const size_t num_x_bins = 300;
  const size_t num_mu_bins = 100;

  Mesh x_mesh{0, 3.0, num_x_bins};
  Mesh mu_mesh{-1.0, 1.0, num_mu_bins};

  Material mat1{1.6422722822026428e-01, 100.0};
  Cell cell1{0.0, 0.75, mat1, nullptr, nullptr};

  Material mat2{3.6612717220096727e+00, 50.0};
  Cell cell2{0.75, 3.0, mat2};
  cell2.left = &cell1;
  cell1.right = &cell2;

  auto tally = HistogramTally<num_x_bins, num_mu_bins>(x_mesh, mu_mesh);

  const size_t num_trials = 1e5;
  double x0 = 0.0;
  double mu0 = 1.0;
  for (double ds: {0.02, 0.01, 0.005}) {
    tally.Reset();
    std::print("Starting DS: {0}\n", ds);
    for (auto i = 0; i < num_trials; i++) {
      Particle p(x0, mu0, &cell1);
      do {
        p.Update(ds, tally);
      } while (p.alive);
    }
    std::print("\tFinished Solve\n");

    auto data_name = "ds_" + std::to_string(ds);
    tally.WriteOut(data_name, num_trials);
    std::print("\tFinished Writeout\n");
  }
}