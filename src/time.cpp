/*
  Copyright (C) 2021 Matt Ginsberg

  This file is intended to be used in conjunction with Stockfish, a
  UCI chess playing engine derived from Glaurung 2.1.  It is not,
  however, part of Stockfish and is licensed separately.

  This particular file is licensed not under the Gnu Public License,
  but instead under the Creative Commons Attribution-NonCommercial
  2.0 Generic (CC BY-NC 2.0) license.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  CC BY-NC 2.0 license for more details.
*/

#include "position.h"
#include "search.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cmath>

static const unsigned ERR = 1e7;

using namespace std;
using namespace Search;

Depth adjust_extension(Depth, const Position &) {
  return 0;
}

// Nodes, predictors, and models are taken from the scikit-learn code.

struct node {
  double        value;
  uint32_t      field;
  double        cutoff;
  bool          nan_go_left;
  uint32_t      left, right;
  bool          fringe;

  node() { }
};

struct predictor : vector<node> {
  predictor() { }
  double increment(const vector<int> &data) const;
};

// Regression model

struct regressor {
  double             base;
  vector<predictor>  predictors;

  regressor() : base(0) { }
  regressor(const string &modelfile);
  double operator[](const vector<int> &data) const;
};

// A classification model is actually just a vector of regression
// models, along with the class names.

struct classifier : vector<regressor> {
  classifier() { }
  classifier(const string &modelfile);
  vector<double> operator[](const vector<int> &data) const;
};

// ... or a classification.

struct classification : vector<double> {
  classification(const vector<double> &vec = vector<double>()) :
    vector<double>(vec) { }
  double                operator[](int idx) const {
    return (idx < 0)? 0 : vector<double>::operator[](idx);
  }
};

istream &operator>>(istream &is, node &n)
{
  string cut;
  is >> n.value >> n.field >> cut >> n.nan_go_left >> n.left >> n.right
     >> n.fringe;
  if (cut == "inf") n.cutoff = ERR;
  else n.cutoff = stod(cut);
  return is;
}

istream &operator>>(istream &is, predictor &p)
{
  unsigned i;
  is >> i;
  p.resize(i);
  for (auto &n : p) is >> n;
  return is;
}

istream &operator>>(istream &is, regressor &m)
{
  unsigned junk, i;
  is >> junk >> m.base >> i;
  m.predictors.resize(i);
  for (auto &p : m.predictors) is >> p;
  return is;
}

// Read a classification model.

istream &operator>>(istream &is, classifier &m)
{
  unsigned num_classes;
  is >> num_classes;
  m.resize(num_classes);
  for (auto &x : m) is >> x.base;
  unsigned npred;
  is >> npred;
  for (auto &x : m) x.predictors.resize(npred);
  for (unsigned i = 0 ; i < npred ; ++i)
    for (auto &x : m) is >> x.predictors[i];
  return is;
}

// Read a regressor, which means you need a bunch of header
// information, a one-hot encoder, and an actual model.

regressor::regressor(const string &modelfile) : regressor()
{
  ifstream inf(modelfile);
  inf >> *this;
}

// Ditto for a classifier

classifier::classifier(const string &modelfile) : classifier()
{
  ifstream inf(modelfile);
  inf >> *this;
}

// A model evaluates data by adding all of the individual predictions
// to a base value.

double regressor::operator[](const vector<int> &data) const
{
  double b = base;
  for (auto &p : predictors) b += p.increment(data);
  return b;
}

// An individual prediction comes from traversing the tree.

double predictor::increment(const vector<int> &data) const
{
  unsigned idx = 0;
  while (!(*this)[idx].fringe) {
    double d = data[(*this)[idx].field];
    unsigned left = (*this)[idx].left;
    unsigned right = (*this)[idx].right;
    idx = (d <= (*this)[idx].cutoff)? left : right;
  }
  return (*this)[idx].value;
}

// Convert a 2-class classifier to probabilities.

double make_prob(double x) { return 1 / (1 + exp(x)); }

// Classification model, copied over from scikit-learn

vector<double> classifier::operator[](const vector<int> &data) const
{
  vector<double> ans;
  for (auto &m : *this) ans.push_back(m[data]);
  if (ans.size() == 1) {
    ans[0] = make_prob(ans[0]);
    ans.push_back(1 - ans[0]);
  }
  else {
    double sum = 0;
    for (double d : ans) sum += exp(d);
    double lse = log(sum);
    for (double &d : ans) d = exp(d - lse);
  }
  return ans;
}

double step13_prunable(const vector<int> &data)
{
  static bool model_read = false;
  static classifier model;
  if (!model_read) {
    model = classifier("/home/ginsberg/chess/data/step13.txt");
    model_read = true;
  }
  return model[data][1];
}

void test_boost()
{
  vector<int> sample(19);
  ifstream inf("../../data/step13.input");
  string line;
  getline(inf,line);
  for (auto &i : sample) inf >> i;
  cout << "sample:";
  for (auto i : sample) cout << ' ' << i;
  cout << endl;
  classifier c("../../data/step13.txt");
  cout << "classifier read" << endl;
  vector<double> ans = c[sample];
  for (auto &d : ans) cout << d << ' ';
  cout << endl;
}
