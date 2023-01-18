
#include "mujoco/mujoco.h"

#include <Eigen/Dense>

using namespace Eigen;

/*
Add to XML file:

    Inside <asset>
        <hfield name='hf1' nrow='2' ncol='6' size="2.5 2.5 .3 .99"/> 


    Inside <worldbody>
        <body name='floor' pos='0 0 0'>
            <geom name='hfield1' pos='2 0 -0.00' hfield='hf1'  type='hfield'      condim='3' conaffinity='15'/>
        </body>



*/

bool create_ramp(mjModel* m_, double degrees, double length, double base_z)
{
    double height = length*(2/5.) * sin(degrees * M_PI/180.0);

	Eigen::MatrixXd terrain(2, 6);
    terrain(0) = 0;
    terrain(1) = 0;
    terrain(2) = 0;
    terrain(3) = 0;
    terrain(4) = 0;
    terrain(5) = 0;
    terrain(6) = height/2.;
    terrain(7) = height/2.;
    terrain(8) = height;
    terrain(9) = height;
    terrain(10) = height;
    terrain(11) = height;

    int id = 0;
    MatrixXd& map = terrain;
    double radius_x = length;
    double radius_y = 1; 

    printf("%f %f %f \n", 0, height/2., height);

    //copied function
    const double max_height = map.maxCoeff();

	if (max_height > 1e-5)
	{
		map = map * (1.0 / max_height); // Normalize matrix
	}

	if (id >= m_->nhfield)
	{
        printf("hfield does not exist, create one in the XML file\n");
		return false;
	}
	int nele = m_->hfield_ncol[id] * m_->hfield_nrow[id];
	if (nele != map.size())
	{
		// Incorrect size
        printf("incorrect size. XML: %d, Given: %d \n", nele, map.size());
		return false;
	}

	int h_i = m_->hfield_adr[id];

	// HField data is row-major
	for (int r = 0; r < map.rows(); r++)
	{
		for (int c = 0; c < map.cols(); c++)
		{
			m_->hfield_data[h_i++] = map(r, c);
            printf("%f", map(r, c));
		}
	}

	if (max_height > 0.0)
	{
		m_->hfield_size[id * 4 + 2] = max_height;
	}
	if (radius_x > 0.0 && radius_y > 0.0)
	{
		m_->hfield_size[id * 4 + 0] = radius_x;
		m_->hfield_size[id * 4 + 1] = radius_y;
	}
	if (base_z > 0.0)
	{
		m_->hfield_size[id * 4 + 3] = base_z;
	}

	return true;
}
