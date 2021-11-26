// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ViewerData.h"
#include "ViewerCore.h"

#include "../per_face_normals.h"
#include "../material_colors.h"
#include "../parula.h"
#include "../per_vertex_normals.h"
#include "igl/png/texture_from_png.h"
#include <iostream>
//#include "external/stb/igl_stb_image.h"

// OUR imports
#include <igl/edge_flaps.h>
#include <igl/parallel_for.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/collapse_edge.h>
#include <igl/per_face_normals.h>
#include <Eigen/Core>
#include <limits>
#include <igl/edge_collapse_is_valid.h>
#include <igl/circulation.h>

IGL_INLINE igl::opengl::ViewerData::ViewerData()
: dirty(MeshGL::DIRTY_ALL),
  show_faces(true),
  show_lines(true),
  invert_normals(false),
  show_overlay(true),
  show_overlay_depth(true),
  show_vertid(false),
  show_faceid(false),
  show_texture(false),
  point_size(30),
  line_width(0.5f),
  line_color(0,0,0,1),
  label_color(0,0,0.04,1),
  shininess(35.0f),
  id(-1),
  is_visible(1)
{
  clear();
};

// OUR functions
IGL_INLINE void igl::opengl::ViewerData::init_simplify() {    
    edge_flaps(F, E, EMAP, EF, EI);

    init_quad_costs();
}

IGL_INLINE void igl::opengl::ViewerData::init_quad_costs() {
    // ASSUMPTION: the ith F_normals corresponds to the ith F face
    compute_normals();
    Eigen::MatrixXd planes = F_normals.normalized();
    planes.conservativeResize(F.rows(), 4);

    // calculating the d's of each plane
    for (int i = 0; i < F.rows(); i++) {
        Eigen::VectorXd vertex = V.row(F(i, 0));
        planes(i, 3) = -(planes(i, 0) * vertex(0) + planes(i, 1) * vertex(1) + planes(i, 2) * vertex(2));
    }

    // init the map
    for (int i = 0; i < V.rows(); i++) {
        Q_quad[i] = Eigen::MatrixXd::Zero(4, 4);
    }

    // calculating Q for each vertex
    for (int i = 0; i < F.rows(); i++) {
        Eigen::MatrixXd Kp = planes.row(i).transpose() * planes.row(i);
        Q_quad[F(i, 0)] += Kp;
        Q_quad[F(i, 1)] += Kp;
        Q_quad[F(i, 2)] += Kp;
    }
  
    // calculating the contractions and their costs
    Eigen::Vector4d last_row { 0.0, 0.0, 0.0, 1.0 };
    Eigen::VectorXd optimal_vertex;
    double optimal_cost;
    for (int i = 0; i < E.rows(); i++) {
        Eigen::MatrixXd Q_roof = Q_quad[E(i, 0)] + Q_quad[E(i, 1)];

        Eigen::MatrixXd Q_roof_prime = Q_roof;
        Q_roof_prime.row(3) = last_row;
        Q_roof_prime = Q_roof_prime.inverse(); // inverting
        if (Q_roof_prime(0, 0) != std::numeric_limits<float>::infinity()) { // if there's inf then the matrix is not invertible
            optimal_vertex = Q_roof_prime * last_row; // 4 x 1
            optimal_cost = optimal_vertex.transpose() * Q_roof * optimal_vertex;
            E_cost.push(std::pair<double, int>(optimal_cost, i));
            optimal_vertex.conservativeResize(3, 1);
            contractions[i] = optimal_vertex;
        }
        else { // the matrix is not invertible and we have to choose one of 3 options
            // choosing optimal???

            Eigen::VectorXd vertex1 = V.row(E(i, 0)); // v1
            double v1_cost = vertex1.transpose() * Q_roof * vertex1; // v1 cost

            Eigen::VectorXd vertex2 = V.row(E(i, 1)); // v2
            double v2_cost = vertex2.transpose() * Q_roof * vertex1; // v2 cost

            Eigen::VectorXd vertex12 = (V.row(E(i, 0)) + V.row(E(i, 1))) / 2; //]the middle of v1 and v2
            double v12_cost = vertex12.transpose() * Q_roof * vertex12; // v2 cost

            if (v1_cost < v2_cost) {
                if (v1_cost < v12_cost) {
                    E_cost.push(std::pair<double, int>(v1_cost, i));
                    contractions[i] = vertex1;
                }
                else {
                    E_cost.push(std::pair<double, int>(v12_cost, i));
                    contractions[i] = vertex12;
                }
            }
            else {
                if (v2_cost < v12_cost) {
                    E_cost.push(std::pair<double, int>(v2_cost, i));
                    contractions[i] = vertex2;
                }
                else {
                    E_cost.push(std::pair<double, int>(v12_cost, i));
                    contractions[i] = vertex12;
                }
            }
        }
    }

    /*  Eigen::Matrix2f A;
    A << 9, 6,
        12, 8;

    std::cout << A.inverse() << std::endl;*/

    /* E_cost.push(std::pair<double, int>(200, 1));
     E_cost.push(std::pair<double, int>(0.1, 2));
     E_cost.push(std::pair<double, int>(150, 2));
     std::pair<double, int> top = E_cost.top();
     std::cout << top.first << " " << top.second << std::endl;
     E_cost.pop();
     top = E_cost.top();
     std::cout << top.first << " " << top.second << std::endl;
     E_cost.pop();
     top = E_cost.top();
     std::cout << top.first << " " << top.second << std::endl;*/
}

IGL_INLINE void igl::opengl::ViewerData::simplify_mesh(int edges_to_remove) {
    // setups
    Eigen::MatrixXd V_proc = V;
    Eigen::MatrixXi F_proc = F;

   
    if (!init_costs_flag) {
        C.resize(E.rows(), V_proc.cols());
        
        Eigen::VectorXd costs(E.rows());
        igl::parallel_for(E.rows(), [&](const int e)
            {

                double cost = e;
                Eigen::RowVectorXd p(1, 3);
                shortest_edge_and_midpoint(e, V_proc, F_proc, E, EMAP, EF, EI, cost, p);
                C.row(e) = p;
                costs(e) = cost;

            }, 10000);

        for (int e = 0;e < E.rows();e++)
        {
            std::set<std::pair<double, int> >::iterator ret = Q.insert(std::pair<double, int>(costs(e), e)).first;
            Qit.push_back(ret);
        }

        init_costs_flag = true;
    }
   

    if (!Q.empty())
    {
        bool something_collapsed = false;
        // collapse edge
        for (int j = 0;j < edges_to_remove; j++)
        {
            if (!collapse_edge(shortest_edge_and_midpoint, V_proc, F_proc, E, EMAP, EF, EI, Q, Qit, C))
            {
                break;
            }
            something_collapsed = true;
        }

        if (something_collapsed)
        {
            clear();
            set_mesh(V_proc, F_proc);
            set_face_based(true);
            dirty = 157;
        }
    }
}

IGL_INLINE void igl::opengl::ViewerData::update_costs() {

    // calculating the contractions and their costs
    Eigen::Vector4d last_row{ 0.0, 0.0, 0.0, 1.0 };
    Eigen::VectorXd optimal_vertex;
    double optimal_cost;
    E_cost = std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>> >();
    for (int i = 0; i < E.rows(); i++) {
        Eigen::MatrixXd Q_roof = Q_quad[E(i, 0)] + Q_quad[E(i, 1)];

        Eigen::MatrixXd Q_roof_prime = Q_roof;
        Q_roof_prime.row(3) = last_row;
        Q_roof_prime = Q_roof_prime.inverse(); // inverting
        if (Q_roof_prime(0, 0) != std::numeric_limits<float>::infinity()) { // if there's inf then the matrix is not invertible
            optimal_vertex = Q_roof_prime * last_row; // 4 x 1
            optimal_cost = optimal_vertex.transpose() * Q_roof * optimal_vertex;
            E_cost.push(std::pair<double, int>(optimal_cost, i));
            optimal_vertex.conservativeResize(3, 1);
            contractions[i] = optimal_vertex;
        }
        else { // the matrix is not invertible and we have to choose one of 3 options
            // choosing optimal???

            Eigen::VectorXd vertex1 = V.row(E(i, 0)); // v1
            double v1_cost = vertex1.transpose() * Q_roof * vertex1; // v1 cost

            Eigen::VectorXd vertex2 = V.row(E(i, 1)); // v2
            double v2_cost = vertex2.transpose() * Q_roof * vertex1; // v2 cost

            Eigen::VectorXd vertex12 = (V.row(E(i, 0)) + V.row(E(i, 1))) / 2; //]the middle of v1 and v2
            double v12_cost = vertex12.transpose() * Q_roof * vertex12; // v2 cost

            if (v1_cost < v2_cost) {
                if (v1_cost < v12_cost) {
                    E_cost.push(std::pair<double, int>(v1_cost, i));
                    contractions[i] = vertex1;
                }
                else {
                    E_cost.push(std::pair<double, int>(v12_cost, i));
                    contractions[i] = vertex12;
                }
            }
            else {
                if (v2_cost < v12_cost) {
                    E_cost.push(std::pair<double, int>(v2_cost, i));
                    contractions[i] = vertex2;
                }
                else {
                    E_cost.push(std::pair<double, int>(v12_cost, i));
                    contractions[i] = vertex12;
                }
            }
        }
    }
}

IGL_INLINE void igl::opengl::ViewerData::simplify_mesh_quad_err(int edges_to_remove) {

    for (int i = 0; i < edges_to_remove; i++) {
        if (E_cost.empty()) {
            break;
        }
        std::pair<double, int> lowest_cost = E_cost.top(); // getting the lowest cost (cost, edge)
        E_cost.pop();
        int edge_to_collapse = lowest_cost.second;
       Eigen::VectorXd new_vetrex = contractions[edge_to_collapse];


        const int eflip = E(edge_to_collapse, 0) > E(edge_to_collapse, 1);
        // source and destination
        const int s = eflip ? E(edge_to_collapse, 1) : E(edge_to_collapse, 0);
        const int d = eflip ? E(edge_to_collapse, 0) : E(edge_to_collapse, 1);

        if (!edge_collapse_is_valid(edge_to_collapse, F, E, EMAP, EF, EI))
        {
            i--; // we still want to collapse the correct number of edges
            continue;
        }

        // Important to grab neighbors of d before monkeying with edges
        const std::vector<int> nV2Fd = circulation(edge_to_collapse, !eflip, EMAP, EF, EI);

        // The following implementation strongly relies on s<d
        assert(s < d && "s should be less than d");
        // move source and destination to midpoint
        V.row(s) = new_vetrex;
        V.row(d) = new_vetrex;

        // update edge info
        // for each flap
        const int m = F.rows();
        for (int side = 0;side < 2;side++)
        {
            const int f = EF(edge_to_collapse, side);
            const int v = EI(edge_to_collapse, side);
            const int sign = (eflip == 0 ? 1 : -1) * (1 - 2 * side);
            // next edge emanating from d
            const int e1 = EMAP(f + m * ((v + sign * 1 + 3) % 3));
            // prev edge pointing to s
            const int e2 = EMAP(f + m * ((v + sign * 2 + 3) % 3));
            assert(E(e1, 0) == d || E(e1, 1) == d);
            assert(E(e2, 0) == s || E(e2, 1) == s);
            // face adjacent to f on e1, also incident on d
            const bool flip1 = EF(e1, 1) == f;
            const int f1 = flip1 ? EF(e1, 0) : EF(e1, 1);
            assert(f1 != f);
            assert(F(f1, 0) == d || F(f1, 1) == d || F(f1, 2) == d);
            // across from which vertex of f1 does e1 appear?
            const int v1 = flip1 ? EI(e1, 0) : EI(e1, 1);
            // Kill e1
            E(e1, 0) = IGL_COLLAPSE_EDGE_NULL;
            E(e1, 1) = IGL_COLLAPSE_EDGE_NULL;
            EF(e1, 0) = IGL_COLLAPSE_EDGE_NULL;
            EF(e1, 1) = IGL_COLLAPSE_EDGE_NULL;
            EI(e1, 0) = IGL_COLLAPSE_EDGE_NULL;
            EI(e1, 1) = IGL_COLLAPSE_EDGE_NULL;
            // Kill f
            F(f, 0) = IGL_COLLAPSE_EDGE_NULL;
            F(f, 1) = IGL_COLLAPSE_EDGE_NULL;
            F(f, 2) = IGL_COLLAPSE_EDGE_NULL;
            // map f1's edge on e1 to e2
            assert(EMAP(f1 + m * v1) == e1);
            EMAP(f1 + m * v1) = e2;
            // side opposite f2, the face adjacent to f on e2, also incident on s
            const int opp2 = (EF(e2, 0) == f ? 0 : 1);
            assert(EF(e2, opp2) == f);
            EF(e2, opp2) = f1;
            EI(e2, opp2) = v1;
            // remap e2 from d to s
            E(e2, 0) = E(e2, 0) == d ? s : E(e2, 0);
            E(e2, 1) = E(e2, 1) == d ? s : E(e2, 1);
        }

        // finally, reindex faces and edges incident on d. Do this last so asserts
        // make sense.
        //
        // Could actually skip first and last, since those are always the two
        // collpased faces.
        for (auto f : nV2Fd)
        {
            for (int v = 0;v < 3;v++)
            {
                if (F(f, v) == d)
                {
                    const int flip1 = (EF(EMAP(f + m * ((v + 1) % 3)), 0) == f) ? 1 : 0;
                    const int flip2 = (EF(EMAP(f + m * ((v + 2) % 3)), 0) == f) ? 0 : 1;
                    assert(
                        E(EMAP(f + m * ((v + 1) % 3)), flip1) == d ||
                        E(EMAP(f + m * ((v + 1) % 3)), flip1) == s);
                    E(EMAP(f + m * ((v + 1) % 3)), flip1) = s;
                    assert(
                        E(EMAP(f + m * ((v + 2) % 3)), flip2) == d ||
                        E(EMAP(f + m * ((v + 2) % 3)), flip2) == s);
                    E(EMAP(f + m * ((v + 2) % 3)), flip2) = s;
                    F(f, v) = s;
                    break;
                }
            }
        }
        // Finally, "remove" this edge and its information
           E(edge_to_collapse, 0) = IGL_COLLAPSE_EDGE_NULL;
           E(edge_to_collapse, 1) = IGL_COLLAPSE_EDGE_NULL;
           EF(edge_to_collapse, 0) = IGL_COLLAPSE_EDGE_NULL;
           EF(edge_to_collapse, 1) = IGL_COLLAPSE_EDGE_NULL;
           EI(edge_to_collapse, 0) = IGL_COLLAPSE_EDGE_NULL;
           EI(edge_to_collapse, 1) = IGL_COLLAPSE_EDGE_NULL;





























        

  //      if (!edge_collapse_is_valid(edge_to_collapse, F, E, EMAP, EF, EI)) { // if can't collapse the current edge then move to the next one
  //          i--; // we still want to collapse the correct number of edges
  //          continue;
  //      }
  //      
  //      const int eflip = E(edge_to_collapse, 0) > E(edge_to_collapse, 1);
  //      // source and destination
  //      const int v1 = eflip ? E(edge_to_collapse, 1) : E(edge_to_collapse, 0);
  //      const int v2 = eflip ? E(edge_to_collapse, 0) : E(edge_to_collapse, 1);
  //      int f1_to_collapse = EF(edge_to_collapse, 0);
  //      int f2_to_collapse = EF(edge_to_collapse, 1);
  //      Eigen::RowVectorXd new_vetrex = contractions[edge_to_collapse].transpose();

  //      // setting up the new vertex
  //      Q_quad[v1] = Q_quad[v1] + Q_quad[v2];
  //      V.row(v1) = new_vetrex;
  //      V.row(v2) = new_vetrex;


  //      // update edge info
  //// for each flap
  //      const int m = F.rows();
  //      for (int side = 0;side < 2;side++)
  //      {
  //          const int f = EF(edge_to_collapse, side);
  //          const int v = EI(edge_to_collapse, side);
  //          const int sign = (eflip == 0 ? 1 : -1) * (1 - 2 * side);
  //          // next edge emanating from d
  //          const int e1 = EMAP(f + m * ((v + sign * 1 + 3) % 3));
  //          // prev edge pointing to s
  //          const int e2 = EMAP(f + m * ((v + sign * 2 + 3) % 3));
  //          assert(E(e1, 0) == v2 || E(e1, 1) == v2);
  //          assert(E(e2, 0) == v1 || E(e2, 1) == v1);
  //          // face adjacent to f on e1, also incident on d
  //          const bool flip1 = EF(e1, 1) == f;
  //          const int f1 = flip1 ? EF(e1, 0) : EF(e1, 1);
  //          assert(f1 != f);
  //          assert(F(f1, 0) == v2 || F(f1, 1) == v2 || F(f1, 2) == v2);
  //          // across from which vertex of f1 does e1 appear?
  //          const int v1 = flip1 ? EI(e1, 0) : EI(e1, 1);
  //          // Kill e1
  //          E(e1, 0) = IGL_COLLAPSE_EDGE_NULL;
  //          E(e1, 1) = IGL_COLLAPSE_EDGE_NULL;
  //          EF(e1, 0) = IGL_COLLAPSE_EDGE_NULL;
  //          EF(e1, 1) = IGL_COLLAPSE_EDGE_NULL;
  //          EI(e1, 0) = IGL_COLLAPSE_EDGE_NULL;
  //          EI(e1, 1) = IGL_COLLAPSE_EDGE_NULL;
  //          // Kill f
  //          F(f, 0) = IGL_COLLAPSE_EDGE_NULL;
  //          F(f, 1) = IGL_COLLAPSE_EDGE_NULL;
  //          F(f, 2) = IGL_COLLAPSE_EDGE_NULL;
  //          // map f1's edge on e1 to e2
  //          assert(EMAP(f1 + m * v1) == e1);
  //          EMAP(f1 + m * v1) = e2;
  //          // side opposite f2, the face adjacent to f on e2, also incident on s
  //          const int opp2 = (EF(e2, 0) == f ? 0 : 1);
  //          assert(EF(e2, opp2) == f);
  //          EF(e2, opp2) = f1;
  //          EI(e2, opp2) = v1;
  //          // remap e2 from d to s
  //          E(e2, 0) = E(e2, 0) == v2 ? v1 : E(e2, 0);
  //          E(e2, 1) = E(e2, 1) == v2 ? v1 : E(e2, 1);
  //      }


  //      // connecting the faces from v2 to v1
  //      for (int j = 0; j < F.rows(); j++) {
  //          if (F(j, 0) == v2) {
  //              F(j, 0) = v1;
  //          }

  //          if (F(j, 1) == v2) {
  //              F(j, 1) = v1;
  //          }

  //          if (F(j, 2) == v2) {
  //              F(j, 2) = v1;
  //          }
  //      }




        // setting up the mesh again after the changes
        Eigen::MatrixXd V_temp = V;
        Eigen::MatrixXi F_temp = F;

        clear();
        set_mesh(V_temp, F_temp);
        set_face_based(true);
        dirty = 157;

        if (i % 10 == 0) {
            update_costs();
        }

        std::cout << "edge " << edge_to_collapse << ", cost = " << lowest_cost.first << ", new v position (" << new_vetrex << ")" << std::endl;







       // // collapsing faces
       //// first face
       // F(f1_to_collapse, 0) = IGL_COLLAPSE_EDGE_NULL;
       // F(f1_to_collapse, 1) = IGL_COLLAPSE_EDGE_NULL;
       // F(f1_to_collapse, 2) = IGL_COLLAPSE_EDGE_NULL;
       // //second face
       // F(f2_to_collapse, 0) = IGL_COLLAPSE_EDGE_NULL;
       // F(f2_to_collapse, 1) = IGL_COLLAPSE_EDGE_NULL;
       // F(f2_to_collapse, 2) = IGL_COLLAPSE_EDGE_NULL;

       // // collapsing the edge
       // E(edge_to_collapse, 0) = IGL_COLLAPSE_EDGE_NULL;
       // E(edge_to_collapse, 1) = IGL_COLLAPSE_EDGE_NULL;
       // EF(edge_to_collapse, 0) = IGL_COLLAPSE_EDGE_NULL;
       // EF(edge_to_collapse, 1) = IGL_COLLAPSE_EDGE_NULL;
       // EI(edge_to_collapse, 0) = IGL_COLLAPSE_EDGE_NULL;
       // EI(edge_to_collapse, 1) = IGL_COLLAPSE_EDGE_NULL;












        
        
        

        

        







































        //std::cout << "EDGE to COLLAPSE " << edge_to_collapse << std::endl;
        //std::cout << "E ROWS: " << E.rows() << std::endl;
        //std::cout << "F rows: " << F.rows() << std::endl;
        //std::cout << "EF  RoWS  " << EF.rows() << std::endl;
        //std::cout << "edge 0 faces: " << EF.row(0) << std::endl;
        //std::cout << "f1 to COLLAPSE " << f1_to_collapse << std::endl;
        //std::cout << "f2 to COLLAPSE " << f2_to_collapse << std::endl;
        //std::cout << "vertex of the edge: " << E.row(edge_to_collapse) << std::endl;
        //std::cout << "face1 to collapse " << F.row(f1_to_collapse) << std::endl;
        //std::cout << "face2 to collapse " << F.row(f2_to_collapse) << std::endl;
        // 
        //// contraction
        //// setting both of v1 to v2 to the new contraption
        ////Q_quad[v1] = Q_quad[v1] + Q_quad[v2];
        


        //// getting the neibors of v1
        //std::set<int> v1_neighbors;
        //for (int j = 0; j < E.rows(); j++) {
        //    if (E(j, 0) == v1) {
        //        v1_neighbors.insert(E(j, 1));
        //    }

        //    if (E(j, 1) == v1) {
        //        v1_neighbors.insert(E(j, 0));
        //    }
        //}

        ////connecting all of v2 edges to v1
        //for (int j = 0; j < E.rows(); j++) {
        //    if (E(j, 0) == v2) {
        //        
        //        if (v1_neighbors.find(E(j, 1)) == v1_neighbors.end()) { // if the other vertex is not already a neighbor of v1 then change to v1
        //            E(j, 0) = v1;
        //        }
        //        else { // collapsing the edge if it's already part of v1
        //            E(j, 0) = IGL_COLLAPSE_EDGE_NULL;
        //            E(j, 1) = IGL_COLLAPSE_EDGE_NULL;
        //        }
        //    }

        //    if (E(j, 1) == v2) {
        //        if (v1_neighbors.find(E(j, 0)) == v1_neighbors.end()) { // if the other vertex is not already a neighbor of v1 then change to v1
        //            E(j, 1) = v1;
        //        }
        //        else { // collapsing the edge if it's already part of v1
        //            E(j, 0) = IGL_COLLAPSE_EDGE_NULL;
        //            E(j, 1) = IGL_COLLAPSE_EDGE_NULL;
        //        }
        //    }
        //}

        //for (int j = 0; j < F.rows(); j++) {
        //    if (F(j, 0) == v2) {
        //        F(j, 0) = v1;
        //    }

        //    if (F(j, 1) == v2) {
        //        F(j, 1) = v1;
        //    }

        //    if (F(j, 2) == v2) {
        //        F(j, 2) = v1;
        //    }
        //}
        //
        //// removing v2
        //// ASSUMPTION: not sure about this
        ///*V(v2, 0) = 0;
        //V(v2, 1) = 0;
        //V(v2, 2) = 0;*/
        //
        //// ASSUMPTION: NOT SURE ABOUT THIS
        //// every 10 iterations recalculated costs 
        ///*if (i % 10 == 0) {
        //    init_quad_costs();
        //}*/

        //Eigen::MatrixXd V_temp = V;
        //Eigen::MatrixXi F_temp = F;

        //clear();
        //set_mesh(V_temp, F_temp);
        //set_face_based(true);
        //dirty = 157;
        //

        //std::cout << "edge " << edge_to_collapse << ", cost = " << lowest_cost.first << ", new v position (" << new_vetrex << ")" << std::endl;
        //edge_flaps(F, E, EMAP, EF, EI);
        //init_quad_costs();
        //std::cout << "hellothere82\n" << std::endl;
    }

    //edge_flaps(F, E, EMAP, EF, EI);
     //std::cout << "hellothere88\n" << std::endl;


    //init_quad_costs();
    update_costs();
}

IGL_INLINE void igl::opengl::ViewerData::set_face_based(bool newvalue)
{
  if (face_based != newvalue)
  {
    face_based = newvalue;
    dirty = MeshGL::DIRTY_ALL;
  }
}

// Helpers that draws the most common meshes
IGL_INLINE void igl::opengl::ViewerData::set_mesh(
    const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F)
{
  using namespace std;

  Eigen::MatrixXd V_temp;

  // If V only has two columns, pad with a column of zeros
  if (_V.cols() == 2)
  {
    V_temp = Eigen::MatrixXd::Zero(_V.rows(),3);
    V_temp.block(0,0,_V.rows(),2) = _V;
  }
  else
    V_temp = _V;

  if (V.rows() == 0 && F.rows() == 0)
  {
    V = V_temp;
    F = _F;

    compute_normals();
    uniform_colors(
      Eigen::Vector3d(GOLD_AMBIENT[0], GOLD_AMBIENT[1], GOLD_AMBIENT[2]),
      Eigen::Vector3d(GOLD_DIFFUSE[0], GOLD_DIFFUSE[1], GOLD_DIFFUSE[2]),
      Eigen::Vector3d(GOLD_SPECULAR[0], GOLD_SPECULAR[1], GOLD_SPECULAR[2]));
	image_texture("D:/UniversityAssiments/Animation/CleanAssignment1/EngineForAnimationCourse/tutorial/textures/snake1.png");
//    grid_texture();
  }
  else
  {
    if (_V.rows() == V.rows() && _F.rows() == F.rows())
    {
      V = V_temp;
      F = _F;
    }
    else
      cerr << "ERROR (set_mesh): The new mesh has a different number of vertices/faces. Please clear the mesh before plotting."<<endl;
  }
  dirty |= MeshGL::DIRTY_FACE | MeshGL::DIRTY_POSITION;

  // our addition to the function
  if (!init_ds_flag) {
      init_simplify(); 
      init_ds_flag = true;
  }
}

IGL_INLINE void igl::opengl::ViewerData::set_vertices(const Eigen::MatrixXd& _V)
{
  V = _V;
  assert(F.size() == 0 || F.maxCoeff() < V.rows());
  dirty |= MeshGL::DIRTY_POSITION;
}

IGL_INLINE void igl::opengl::ViewerData::set_normals(const Eigen::MatrixXd& N)
{
  using namespace std;
  if (N.rows() == V.rows())
  {
    set_face_based(false);
    V_normals = N;
  }
  else if (N.rows() == F.rows() || N.rows() == F.rows()*3)
  {
    set_face_based(true);
    F_normals = N;
  }
  else
    cerr << "ERROR (set_normals): Please provide a normal per face, per corner or per vertex."<<endl;
  dirty |= MeshGL::DIRTY_NORMAL;
}

IGL_INLINE void igl::opengl::ViewerData::set_visible(bool value, unsigned int core_id /*= 1*/)
{
  if (value)
    is_visible |= core_id;
  else
  is_visible &= ~core_id;
}

//IGL_INLINE void igl::opengl::ViewerData::copy_options(const ViewerCore &from, const ViewerCore &to)
//{
//  to.set(show_overlay      , from.is_set(show_overlay)      );
//  to.set(show_overlay_depth, from.is_set(show_overlay_depth));
//  to.set(show_texture      , from.is_set(show_texture)      );
//  to.set(show_faces        , from.is_set(show_faces)        );
//  to.set(show_lines        , from.is_set(show_lines)        );
//}

IGL_INLINE void igl::opengl::ViewerData::set_colors(const Eigen::MatrixXd &C)
{
  using namespace std;
  using namespace Eigen;
  if(C.rows()>0 && C.cols() == 1)
  {
    Eigen::MatrixXd C3;
    igl::parula(C,true,C3);
    return set_colors(C3);
  }
  // Ambient color should be darker color
  const auto ambient = [](const MatrixXd & C)->MatrixXd
  {
    MatrixXd T = 0.1*C;
    T.col(3) = C.col(3);
    return T;
  };
  // Specular color should be a less saturated and darker color: dampened
  // highlights
  const auto specular = [](const MatrixXd & C)->MatrixXd
  {
    const double grey = 0.3;
    MatrixXd T = grey+0.1*(C.array()-grey);
    T.col(3) = C.col(3);
    return T;
  };
  if (C.rows() == 1)
  {
    for (unsigned i=0;i<V_material_diffuse.rows();++i)
    {
      if (C.cols() == 3)
        V_material_diffuse.row(i) << C.row(0),1;
      else if (C.cols() == 4)
        V_material_diffuse.row(i) << C.row(0);
    }
    V_material_ambient = ambient(V_material_diffuse);
    V_material_specular = specular(V_material_diffuse);

    for (unsigned i=0;i<F_material_diffuse.rows();++i)
    {
      if (C.cols() == 3)
        F_material_diffuse.row(i) << C.row(0),1;
      else if (C.cols() == 4)
        F_material_diffuse.row(i) << C.row(0);
    }
    F_material_ambient = ambient(F_material_diffuse);
    F_material_specular = specular(F_material_diffuse);
  }
  else if (C.rows() == V.rows())
  {
    set_face_based(false);
    for (unsigned i=0;i<V_material_diffuse.rows();++i)
    {
      if (C.cols() == 3)
        V_material_diffuse.row(i) << C.row(i), 1;
      else if (C.cols() == 4)
        V_material_diffuse.row(i) << C.row(i);
    }
    V_material_ambient = ambient(V_material_diffuse);
    V_material_specular = specular(V_material_diffuse);
  }
  else if (C.rows() == F.rows())
  {
    set_face_based(true);
    for (unsigned i=0;i<F_material_diffuse.rows();++i)
    {
      if (C.cols() == 3)
        F_material_diffuse.row(i) << C.row(i), 1;
      else if (C.cols() == 4)
        F_material_diffuse.row(i) << C.row(i);
    }
    F_material_ambient = ambient(F_material_diffuse);
    F_material_specular = specular(F_material_diffuse);
  }
  else
    cerr << "ERROR (set_colors): Please provide a single color, or a color per face or per vertex."<<endl;
  dirty |= MeshGL::DIRTY_DIFFUSE;

}

IGL_INLINE void igl::opengl::ViewerData::set_uv(const Eigen::MatrixXd& UV)
{
  using namespace std;
  if (UV.rows() == V.rows())
  {
    set_face_based(false);
    V_uv = UV;
  }
  else
    cerr << "ERROR (set_UV): Please provide uv per vertex."<<endl;;
  dirty |= MeshGL::DIRTY_UV;
}

IGL_INLINE void igl::opengl::ViewerData::set_uv(const Eigen::MatrixXd& UV_V, const Eigen::MatrixXi& UV_F)
{
  set_face_based(true);
  V_uv = UV_V.block(0,0,UV_V.rows(),2);
  F_uv = UV_F;
  dirty |= MeshGL::DIRTY_UV;
}

IGL_INLINE void igl::opengl::ViewerData::set_texture(
  const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& R,
  const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& G,
  const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& B)
{
  texture_R = R;
  texture_G = G;
  texture_B = B;
  texture_A = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>::Constant(R.rows(),R.cols(),255);
  dirty |= MeshGL::DIRTY_TEXTURE;
}

IGL_INLINE void igl::opengl::ViewerData::set_texture(
  const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& R,
  const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& G,
  const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& B,
  const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& A)
{
  texture_R = R;
  texture_G = G;
  texture_B = B;
  texture_A = A;
  dirty |= MeshGL::DIRTY_TEXTURE;
}

IGL_INLINE void igl::opengl::ViewerData::set_points(
  const Eigen::MatrixXd& P,
  const Eigen::MatrixXd& C)
{
  // clear existing points
  points.resize(0,0);
  add_points(P,C);
}

IGL_INLINE void igl::opengl::ViewerData::add_points(const Eigen::MatrixXd& P,  const Eigen::MatrixXd& C)
{
  Eigen::MatrixXd P_temp;

  // If P only has two columns, pad with a column of zeros
  if (P.cols() == 2)
  {
    P_temp = Eigen::MatrixXd::Zero(P.rows(),3);
    P_temp.block(0,0,P.rows(),2) = P;
  }
  else
    P_temp = P;

  int lastid = points.rows();
  points.conservativeResize(points.rows() + P_temp.rows(),6);
  for (unsigned i=0; i<P_temp.rows(); ++i)
    points.row(lastid+i) << P_temp.row(i), i<C.rows() ? C.row(i) : C.row(C.rows()-1);

  dirty |= MeshGL::DIRTY_OVERLAY_POINTS;
}

IGL_INLINE void igl::opengl::ViewerData::set_edges(
  const Eigen::MatrixXd& P,
  const Eigen::MatrixXi& E,
  const Eigen::MatrixXd& C)
{
  using namespace Eigen;
  lines.resize(E.rows(),9);
  assert(C.cols() == 3);
  for(int e = 0;e<E.rows();e++)
  {
    RowVector3d color;
    if(C.size() == 3)
    {
      color<<C;
    }else if(C.rows() == E.rows())
    {
      color<<C.row(e);
    }
    lines.row(e)<< P.row(E(e,0)), P.row(E(e,1)), color;
  }
  dirty |= MeshGL::DIRTY_OVERLAY_LINES;
}

IGL_INLINE void igl::opengl::ViewerData::add_edges(const Eigen::MatrixXd& P1, const Eigen::MatrixXd& P2, const Eigen::MatrixXd& C)
{
  Eigen::MatrixXd P1_temp,P2_temp;

  // If P1 only has two columns, pad with a column of zeros
  if (P1.cols() == 2)
  {
    P1_temp = Eigen::MatrixXd::Zero(P1.rows(),3);
    P1_temp.block(0,0,P1.rows(),2) = P1;
    P2_temp = Eigen::MatrixXd::Zero(P2.rows(),3);
    P2_temp.block(0,0,P2.rows(),2) = P2;
  }
  else
  {
    P1_temp = P1;
    P2_temp = P2;
  }

  int lastid = lines.rows();
  lines.conservativeResize(lines.rows() + P1_temp.rows(),9);
  for (unsigned i=0; i<P1_temp.rows(); ++i)
    lines.row(lastid+i) << P1_temp.row(i), P2_temp.row(i), i<C.rows() ? C.row(i) : C.row(C.rows()-1);

  dirty |= MeshGL::DIRTY_OVERLAY_LINES;
}

IGL_INLINE void igl::opengl::ViewerData::add_label(const Eigen::VectorXd& P,  const std::string& str)
{
  Eigen::RowVectorXd P_temp;

  // If P only has two columns, pad with a column of zeros
  if (P.size() == 2)
  {
    P_temp = Eigen::RowVectorXd::Zero(3);
    P_temp << P.transpose(), 0;
  }
  else
    P_temp = P;

  int lastid = labels_positions.rows();
  labels_positions.conservativeResize(lastid+1, 3);
  labels_positions.row(lastid) = P_temp;
  labels_strings.push_back(str);
}

IGL_INLINE void igl::opengl::ViewerData::clear_labels()
{
  labels_positions.resize(0,3);
  labels_strings.clear();
}

IGL_INLINE void igl::opengl::ViewerData::clear()
{
  V                       = Eigen::MatrixXd (0,3);
  F                       = Eigen::MatrixXi (0,3);

  F_material_ambient      = Eigen::MatrixXd (0,4);
  F_material_diffuse      = Eigen::MatrixXd (0,4);
  F_material_specular     = Eigen::MatrixXd (0,4);

  V_material_ambient      = Eigen::MatrixXd (0,4);
  V_material_diffuse      = Eigen::MatrixXd (0,4);
  V_material_specular     = Eigen::MatrixXd (0,4);

  F_normals               = Eigen::MatrixXd (0,3);
  V_normals               = Eigen::MatrixXd (0,3);

  V_uv                    = Eigen::MatrixXd (0,2);
  F_uv                    = Eigen::MatrixXi (0,3);

  lines                   = Eigen::MatrixXd (0,9);
  points                  = Eigen::MatrixXd (0,6);
  labels_positions        = Eigen::MatrixXd (0,3);
  labels_strings.clear();

  face_based = false;
}

IGL_INLINE void igl::opengl::ViewerData::compute_normals()
{
  igl::per_face_normals(V, F, F_normals);
  igl::per_vertex_normals(V, F, F_normals, V_normals);
  dirty |= MeshGL::DIRTY_NORMAL;
}

IGL_INLINE void igl::opengl::ViewerData::uniform_colors(
  const Eigen::Vector3d& ambient,
  const Eigen::Vector3d& diffuse,
  const Eigen::Vector3d& specular)
{
  Eigen::Vector4d ambient4;
  Eigen::Vector4d diffuse4;
  Eigen::Vector4d specular4;

  ambient4 << ambient, 1;
  diffuse4 << diffuse, 1;
  specular4 << specular, 1;

  uniform_colors(ambient4,diffuse4,specular4);
}

IGL_INLINE void igl::opengl::ViewerData::uniform_colors(
  const Eigen::Vector4d& ambient,
  const Eigen::Vector4d& diffuse,
  const Eigen::Vector4d& specular)
{
  V_material_ambient.resize(V.rows(),4);
  V_material_diffuse.resize(V.rows(),4);
  V_material_specular.resize(V.rows(),4);

  for (unsigned i=0; i<V.rows();++i)
  {
    V_material_ambient.row(i) = ambient;
    V_material_diffuse.row(i) = diffuse;
    V_material_specular.row(i) = specular;
  }

  F_material_ambient.resize(F.rows(),4);
  F_material_diffuse.resize(F.rows(),4);
  F_material_specular.resize(F.rows(),4);

  for (unsigned i=0; i<F.rows();++i)
  {
    F_material_ambient.row(i) = ambient;
    F_material_diffuse.row(i) = diffuse;
    F_material_specular.row(i) = specular;
  }
  dirty |= MeshGL::DIRTY_SPECULAR | MeshGL::DIRTY_DIFFUSE | MeshGL::DIRTY_AMBIENT;
}

IGL_INLINE void igl::opengl::ViewerData::image_texture(const std::string fileName)
{
	//unsigned int texId;
	//if (igl::png::texture_from_png(fileName, false, texId))
	if(igl::png::texture_from_png(fileName,texture_R, texture_G, texture_B, texture_A))
	
		dirty |= MeshGL::DIRTY_TEXTURE;
	else
		std::cout<<"can't open texture file"<<std::endl;



}

IGL_INLINE void igl::opengl::ViewerData::grid_texture()
{
  // Don't do anything for an empty mesh
  if(V.rows() == 0)
  {
    V_uv.resize(V.rows(),2);
    return;
  }
  if (V_uv.rows() == 0)
  {
    V_uv = V.block(0, 0, V.rows(), 2);
    V_uv.col(0) = V_uv.col(0).array() - V_uv.col(0).minCoeff();
    V_uv.col(0) = V_uv.col(0).array() / V_uv.col(0).maxCoeff();
    V_uv.col(1) = V_uv.col(1).array() - V_uv.col(1).minCoeff();
    V_uv.col(1) = V_uv.col(1).array() / V_uv.col(1).maxCoeff();
    V_uv = V_uv.array() * 10;
    dirty |= MeshGL::DIRTY_TEXTURE;
  }

  unsigned size = 4;
  unsigned size2 = size/2;
  texture_R.resize(size, size);
  for (unsigned i=0; i<size; ++i)
  {
    for (unsigned j=0; j<size; ++j)
    {
      texture_R(i,j) = 0;
      if ((i<size2 && j<size2) || (i>=size2 && j>=size2))
        texture_R(i,j) = 255;
    }
  }

  texture_G = texture_R;
  texture_B = texture_R;
  texture_A = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>::Constant(texture_R.rows(),texture_R.cols(),255);
  dirty |= MeshGL::DIRTY_TEXTURE;
}

IGL_INLINE void igl::opengl::ViewerData::updateGL(
  const igl::opengl::ViewerData& data,
  const bool invert_normals,
  igl::opengl::MeshGL& meshgl
  )
{
  if (!meshgl.is_initialized)
  {
    meshgl.init();
  }

  bool per_corner_uv = (data.F_uv.rows() == data.F.rows());
  bool per_corner_normals = (data.F_normals.rows() == 3 * data.F.rows());

  meshgl.dirty |= data.dirty;

  // Input:
  //   X  #F by dim quantity
  // Output:
  //   X_vbo  #F*3 by dim scattering per corner
  const auto per_face = [&data](
      const Eigen::MatrixXd & X,
      MeshGL::RowMatrixXf & X_vbo)
  {
    assert(X.cols() == 4);
    X_vbo.resize(data.F.rows()*3,4);
    for (unsigned i=0; i<data.F.rows();++i)
      for (unsigned j=0;j<3;++j)
        X_vbo.row(i*3+j) = X.row(i).cast<float>();
  };

  // Input:
  //   X  #V by dim quantity
  // Output:
  //   X_vbo  #F*3 by dim scattering per corner
  const auto per_corner = [&data](
      const Eigen::MatrixXd & X,
      MeshGL::RowMatrixXf & X_vbo)
  {
    X_vbo.resize(data.F.rows()*3,X.cols());
    for (unsigned i=0; i<data.F.rows();++i)
      for (unsigned j=0;j<3;++j)
        X_vbo.row(i*3+j) = X.row(data.F(i,j)).cast<float>();
  };

  if (!data.face_based)
  {
    if (!(per_corner_uv || per_corner_normals))
    {
      // Vertex positions
      if (meshgl.dirty & MeshGL::DIRTY_POSITION)
        meshgl.V_vbo = data.V.cast<float>();

      // Vertex normals
      if (meshgl.dirty & MeshGL::DIRTY_NORMAL)
      {
        meshgl.V_normals_vbo = data.V_normals.cast<float>();
        if (invert_normals)
          meshgl.V_normals_vbo = -meshgl.V_normals_vbo;
      }

      // Per-vertex material settings
      if (meshgl.dirty & MeshGL::DIRTY_AMBIENT)
        meshgl.V_ambient_vbo = data.V_material_ambient.cast<float>();
      if (meshgl.dirty & MeshGL::DIRTY_DIFFUSE)
        meshgl.V_diffuse_vbo = data.V_material_diffuse.cast<float>();
      if (meshgl.dirty & MeshGL::DIRTY_SPECULAR)
        meshgl.V_specular_vbo = data.V_material_specular.cast<float>();

      // Face indices
      if (meshgl.dirty & MeshGL::DIRTY_FACE)
        meshgl.F_vbo = data.F.cast<unsigned>();

      // Texture coordinates
      if (meshgl.dirty & MeshGL::DIRTY_UV)
      {
        meshgl.V_uv_vbo = data.V_uv.cast<float>();
      }
    }
    else
    {

      // Per vertex properties with per corner UVs
      if (meshgl.dirty & MeshGL::DIRTY_POSITION)
      {
        per_corner(data.V,meshgl.V_vbo);
      }

      if (meshgl.dirty & MeshGL::DIRTY_AMBIENT)
      {
        meshgl.V_ambient_vbo.resize(data.F.rows()*3,4);
        for (unsigned i=0; i<data.F.rows();++i)
          for (unsigned j=0;j<3;++j)
            meshgl.V_ambient_vbo.row(i*3+j) = data.V_material_ambient.row(data.F(i,j)).cast<float>();
      }
      if (meshgl.dirty & MeshGL::DIRTY_DIFFUSE)
      {
        meshgl.V_diffuse_vbo.resize(data.F.rows()*3,4);
        for (unsigned i=0; i<data.F.rows();++i)
          for (unsigned j=0;j<3;++j)
            meshgl.V_diffuse_vbo.row(i*3+j) = data.V_material_diffuse.row(data.F(i,j)).cast<float>();
      }
      if (meshgl.dirty & MeshGL::DIRTY_SPECULAR)
      {
        meshgl.V_specular_vbo.resize(data.F.rows()*3,4);
        for (unsigned i=0; i<data.F.rows();++i)
          for (unsigned j=0;j<3;++j)
            meshgl.V_specular_vbo.row(i*3+j) = data.V_material_specular.row(data.F(i,j)).cast<float>();
      }

      if (meshgl.dirty & MeshGL::DIRTY_NORMAL)
      {
        meshgl.V_normals_vbo.resize(data.F.rows()*3,3);
        for (unsigned i=0; i<data.F.rows();++i)
          for (unsigned j=0;j<3;++j)
            meshgl.V_normals_vbo.row(i*3+j) =
                         per_corner_normals ?
               data.F_normals.row(i*3+j).cast<float>() :
               data.V_normals.row(data.F(i,j)).cast<float>();


        if (invert_normals)
          meshgl.V_normals_vbo = -meshgl.V_normals_vbo;
      }

      if (meshgl.dirty & MeshGL::DIRTY_FACE)
      {
        meshgl.F_vbo.resize(data.F.rows(),3);
        for (unsigned i=0; i<data.F.rows();++i)
          meshgl.F_vbo.row(i) << i*3+0, i*3+1, i*3+2;
      }

      if (meshgl.dirty & MeshGL::DIRTY_UV)
      {
        meshgl.V_uv_vbo.resize(data.F.rows()*3,2);
        for (unsigned i=0; i<data.F.rows();++i)
          for (unsigned j=0;j<3;++j)
            meshgl.V_uv_vbo.row(i*3+j) =
              data.V_uv.row(per_corner_uv ?
                data.F_uv(i,j) : data.F(i,j)).cast<float>();
      }
    }
  }
  else
  {
    if (meshgl.dirty & MeshGL::DIRTY_POSITION)
    {
      per_corner(data.V,meshgl.V_vbo);
    }
    if (meshgl.dirty & MeshGL::DIRTY_AMBIENT)
    {
      per_face(data.F_material_ambient,meshgl.V_ambient_vbo);
    }
    if (meshgl.dirty & MeshGL::DIRTY_DIFFUSE)
    {
      per_face(data.F_material_diffuse,meshgl.V_diffuse_vbo);
    }
    if (meshgl.dirty & MeshGL::DIRTY_SPECULAR)
    {
      per_face(data.F_material_specular,meshgl.V_specular_vbo);
    }

    if (meshgl.dirty & MeshGL::DIRTY_NORMAL)
    {
      meshgl.V_normals_vbo.resize(data.F.rows()*3,3);
      for (unsigned i=0; i<data.F.rows();++i)
        for (unsigned j=0;j<3;++j)
          meshgl.V_normals_vbo.row(i*3+j) =
             per_corner_normals ?
               data.F_normals.row(i*3+j).cast<float>() :
               data.F_normals.row(i).cast<float>();

      if (invert_normals)
        meshgl.V_normals_vbo = -meshgl.V_normals_vbo;
    }

    if (meshgl.dirty & MeshGL::DIRTY_FACE)
    {
      meshgl.F_vbo.resize(data.F.rows(),3);
      for (unsigned i=0; i<data.F.rows();++i)
        meshgl.F_vbo.row(i) << i*3+0, i*3+1, i*3+2;
    }

    if (meshgl.dirty & MeshGL::DIRTY_UV)
    {
        meshgl.V_uv_vbo.resize(data.F.rows()*3,2);
        for (unsigned i=0; i<data.F.rows();++i)
          for (unsigned j=0;j<3;++j)
            meshgl.V_uv_vbo.row(i*3+j) = data.V_uv.row(per_corner_uv ? data.F_uv(i,j) : data.F(i,j)).cast<float>();
    }
  }

  if (meshgl.dirty & MeshGL::DIRTY_TEXTURE)
  {
    meshgl.tex_u = data.texture_R.rows();
    meshgl.tex_v = data.texture_R.cols();
    meshgl.tex.resize(data.texture_R.size()*4);
    for (unsigned i=0;i<data.texture_R.size();++i)
    {
      meshgl.tex(i*4+0) = data.texture_R(i);
      meshgl.tex(i*4+1) = data.texture_G(i);
      meshgl.tex(i*4+2) = data.texture_B(i);
      meshgl.tex(i*4+3) = data.texture_A(i);
    }
  }

  if (meshgl.dirty & MeshGL::DIRTY_OVERLAY_LINES)
  {
    meshgl.lines_V_vbo.resize(data.lines.rows()*2,3);
    meshgl.lines_V_colors_vbo.resize(data.lines.rows()*2,3);
    meshgl.lines_F_vbo.resize(data.lines.rows()*2,1);
    for (unsigned i=0; i<data.lines.rows();++i)
    {
      meshgl.lines_V_vbo.row(2*i+0) = data.lines.block<1, 3>(i, 0).cast<float>();
      meshgl.lines_V_vbo.row(2*i+1) = data.lines.block<1, 3>(i, 3).cast<float>();
      meshgl.lines_V_colors_vbo.row(2*i+0) = data.lines.block<1, 3>(i, 6).cast<float>();
      meshgl.lines_V_colors_vbo.row(2*i+1) = data.lines.block<1, 3>(i, 6).cast<float>();
      meshgl.lines_F_vbo(2*i+0) = 2*i+0;
      meshgl.lines_F_vbo(2*i+1) = 2*i+1;
    }
  }

  if (meshgl.dirty & MeshGL::DIRTY_OVERLAY_POINTS)
  {
    meshgl.points_V_vbo.resize(data.points.rows(),3);
    meshgl.points_V_colors_vbo.resize(data.points.rows(),3);
    meshgl.points_F_vbo.resize(data.points.rows(),1);
    for (unsigned i=0; i<data.points.rows();++i)
    {
      meshgl.points_V_vbo.row(i) = data.points.block<1, 3>(i, 0).cast<float>();
      meshgl.points_V_colors_vbo.row(i) = data.points.block<1, 3>(i, 3).cast<float>();
      meshgl.points_F_vbo(i) = i;
    }
  }
}
