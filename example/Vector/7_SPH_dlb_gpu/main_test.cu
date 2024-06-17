#ifdef __NVCC__

#include "Vector/vector_dist.hpp"

template<typename vector_type, typename NN_type>
__global__ void probe_neighbors(vector_type vd, NN_type NN)
{
    auto a = GET_PARTICLE(vd);
    if (a == 10) {
        auto itn = NN.getNNIteratorBoxSym(a, NN.getCell(vd.getPos(a)));
        //Point<2,double> xp({1.0, 1.0});
        //auto itn = NN.getNNIteratorBox(NN.getCell(xp));
        while (itn.isNext()) {
            auto b = itn.get();
            if (a != b)	{
                vd.template getProp<0>(b) = 1.0;
            }
            ++itn;
        }
    }
}

int main(int argc, char* argv[])
{
	openfpm_init(&argc,&argv);

    size_t sz[2] = {10, 10};
    size_t bc[2] = {NON_PERIODIC, NON_PERIODIC};
    Box<2, double> domain({0, 0}, {2.0, 2.0});
    Ghost<2, double> g(0.1);
    vector_dist_gpu<2, double, aggregate<double>> vd(0, domain, bc, g);

	auto grid_it = vd.getGridIterator(sz);
    while (grid_it.isNext()) {
        vd.add();

        auto key = grid_it.get();
        double x = key.get(0) * grid_it.getSpacing(0);
        double y = key.get(1) * grid_it.getSpacing(1);
        double z = key.get(2) * grid_it.getSpacing(2);

        vd.getLastPos()[0] = x;
        vd.getLastPos()[1] = y;
        vd.getLastPos()[2] = z;

        vd.template getLastProp<0>() = -1.0;
        ++grid_it;
    }

    auto NN_sym = vd.getCellListGPU(0.4, CL_SYMMETRIC);
    auto NN_nonsym = vd.getCellListGPU(0.4, CL_NON_SYMMETRIC);
    vd.updateCellListGPU(NN_sym);
    vd.updateCellListGPU(NN_nonsym);
    NN_sym.debug_deviceToHost();
    NN_nonsym.debug_deviceToHost();
    std::cout << "Number of cells :" << NN_sym.getNCells() << " " << NN_nonsym.getNCells() << std::endl;
    
    std::cout << "Elements distribution across NN_nonsym cells:" << std::endl;
    for (int i = 0; i < NN_nonsym.getNCells(); i++) {
        if (NN_nonsym.getNelements(i) > 0) {
            std::cout << i << ": " << NN_nonsym.getNelements(i) << std::endl;
        }
    }

    std::cout << "Elements distribution across NN_sym cells:" << std::endl;
    for (int i = 0; i < NN_sym.getNCells(); i++) {
        if (NN_sym.getNelements(i) > 0) {
            std::cout << i << ": " << NN_sym.getNelements(i) << std::endl;
        }
    }
    
    auto vd_it_gpu = vd.getDomainIteratorGPU();
    CUDA_LAUNCH(probe_neighbors, vd_it_gpu, vd.toKernel(), NN_sym.toKernel());
    vd.template deviceToHostProp<0>();
    
    auto vd_it = vd.getDomainIterator();
    while (vd_it.isNext()) {
        auto key = vd_it.get();
        std::cout << vd.getPos(key)[0] << " " << vd.getPos(key)[1] << " " << vd.getProp<0>(key) << std::endl;
        ++vd_it;
    }

	openfpm_finalize();
}
 
#else

int main(int argc, char* argv[])
{
        return 0;
}

#endif