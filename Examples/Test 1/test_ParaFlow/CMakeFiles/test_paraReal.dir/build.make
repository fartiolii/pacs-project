# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/Desktop/PacsProject/project/tests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Desktop/PacsProject/project/tests

# Include any dependencies generated for this target.
include CMakeFiles/test_paraReal.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_paraReal.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_paraReal.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_paraReal.dir/flags.make

CMakeFiles/test_paraReal.dir/test_paraReal.cc.o: CMakeFiles/test_paraReal.dir/flags.make
CMakeFiles/test_paraReal.dir/test_paraReal.cc.o: test_paraReal.cc
CMakeFiles/test_paraReal.dir/test_paraReal.cc.o: CMakeFiles/test_paraReal.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/Desktop/PacsProject/project/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_paraReal.dir/test_paraReal.cc.o"
	/usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_paraReal.dir/test_paraReal.cc.o -MF CMakeFiles/test_paraReal.dir/test_paraReal.cc.o.d -o CMakeFiles/test_paraReal.dir/test_paraReal.cc.o -c /home/ubuntu/Desktop/PacsProject/project/tests/test_paraReal.cc

CMakeFiles/test_paraReal.dir/test_paraReal.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_paraReal.dir/test_paraReal.cc.i"
	/usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/Desktop/PacsProject/project/tests/test_paraReal.cc > CMakeFiles/test_paraReal.dir/test_paraReal.cc.i

CMakeFiles/test_paraReal.dir/test_paraReal.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_paraReal.dir/test_paraReal.cc.s"
	/usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/Desktop/PacsProject/project/tests/test_paraReal.cc -o CMakeFiles/test_paraReal.dir/test_paraReal.cc.s

# Object files for target test_paraReal
test_paraReal_OBJECTS = \
"CMakeFiles/test_paraReal.dir/test_paraReal.cc.o"

# External object files for target test_paraReal
test_paraReal_EXTERNAL_OBJECTS =

test_paraReal: CMakeFiles/test_paraReal.dir/test_paraReal.cc.o
test_paraReal: CMakeFiles/test_paraReal.dir/build.make
test_paraReal: /home/ubuntu/Documents/deal.II-v9.5.1/lib/libdeal_II.so.9.5.1
test_paraReal: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libboost_system.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libboost_thread.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libboost_regex.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/librol.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtempus.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libmuelu-adapters.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libmuelu-interface.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libmuelu.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/liblocathyra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/liblocaepetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/liblocalapack.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libloca.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libnoxepetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libnoxlapack.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libnox.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libintrepid2.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libintrepid.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteko.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libstratimikos.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libstratimikosbelos.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libstratimikosamesos2.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libstratimikosaztecoo.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libstratimikosamesos.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libstratimikosml.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libstratimikosifpack.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libanasazitpetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libModeLaplace.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libanasaziepetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libanasazi.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libamesos2.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtacho.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libbelosxpetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libbelostpetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libbelosepetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libbelos.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libml.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libifpack.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libzoltan2.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libpamgen_extras.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libpamgen.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libamesos.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libgaleri-xpetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libgaleri-epetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libaztecoo.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libisorropia.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libxpetra-sup.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libxpetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libthyratpetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libthyraepetraext.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libthyraepetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libthyracore.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtrilinosss.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtpetraext.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtpetrainout.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtpetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libkokkostsqr.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtpetraclassiclinalg.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtpetraclassicnodeapi.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtpetraclassic.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libepetraext.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libtriutils.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libshards.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libzoltan.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libepetra.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libsacado.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/librtop.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libkokkoskernels.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteuchoskokkoscomm.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteuchoskokkoscompat.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteuchosremainder.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteuchosnumerics.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteuchoscomm.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteuchosparameterlist.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteuchosparser.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libteuchoscore.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libkokkosalgorithms.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libkokkoscontainers.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libkokkoscore.so
test_paraReal: /home/ubuntu/Documents/trilinos-release-13-2-0/lib/libgtest.so
test_paraReal: /home/ubuntu/Documents/superlu_dist_5.1.2/lib/libsuperlu_dist.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libumfpack.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libcholmod.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libccolamd.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libcolamd.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libcamd.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libamd.so
test_paraReal: /home/ubuntu/Documents/hdf5-1.12.2/lib/libhdf5.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libdl.a
test_paraReal: /usr/lib/x86_64-linux-gnu/libm.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKBO.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKBool.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKBRep.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKernel.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKFeat.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKFillet.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKG2d.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKG3d.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKGeomAlgo.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKGeomBase.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKHLR.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKIGES.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKMath.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKMesh.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKOffset.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKPrim.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKShHealing.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKSTEP.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKSTEPAttr.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKSTEPBase.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKSTEP209.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKSTL.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKTopAlgo.so
test_paraReal: /home/ubuntu/Documents/oce-OCE-0.18.3/lib/libTKXSBase.so
test_paraReal: /home/ubuntu/Documents/slepc-3.18.3/lib/libslepc.so
test_paraReal: /home/ubuntu/Documents/petsc-3.18.6/lib/libpetsc.so
test_paraReal: /home/ubuntu/Documents/petsc-3.18.6/lib/libHYPRE.so
test_paraReal: /home/ubuntu/Documents/petsc-3.18.6/lib/libdmumps.a
test_paraReal: /home/ubuntu/Documents/petsc-3.18.6/lib/libmumps_common.a
test_paraReal: /home/ubuntu/Documents/petsc-3.18.6/lib/libpord.a
test_paraReal: /home/ubuntu/Documents/petsc-3.18.6/lib/libscalapack.so
test_paraReal: /home/ubuntu/Documents/parmetis-4.0.3/lib/libparmetis.so
test_paraReal: /home/ubuntu/Documents/parmetis-4.0.3/lib/libmetis.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libmpi_usempif08.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libmpi_usempi_ignore_tkr.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libmpi_mpifh.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libmpi.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libopen-rte.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libopen-pal.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libhwloc.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libevent_core.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libevent_pthreads.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libz.so
test_paraReal: /home/ubuntu/Documents/sundials-5.7.0/lib/libsundials_idas.so
test_paraReal: /home/ubuntu/Documents/sundials-5.7.0/lib/libsundials_arkode.so
test_paraReal: /home/ubuntu/Documents/sundials-5.7.0/lib/libsundials_kinsol.so
test_paraReal: /home/ubuntu/Documents/sundials-5.7.0/lib/libsundials_nvecserial.so
test_paraReal: /home/ubuntu/Documents/sundials-5.7.0/lib/libsundials_nvecparallel.so
test_paraReal: /home/ubuntu/Documents/symengine-0.8.1/lib/libsymengine.so.0.8.1
test_paraReal: /usr/lib/x86_64-linux-gnu/libgmp.so
test_paraReal: /usr/lib/x86_64-linux-gnu/liblapack.so
test_paraReal: /usr/lib/x86_64-linux-gnu/libblas.so
test_paraReal: /home/ubuntu/Documents/p4est-2.3.2/FAST/lib/libp4est.so
test_paraReal: /home/ubuntu/Documents/p4est-2.3.2/FAST/lib/libsc.so
test_paraReal: CMakeFiles/test_paraReal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/Desktop/PacsProject/project/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_paraReal"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_paraReal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_paraReal.dir/build: test_paraReal
.PHONY : CMakeFiles/test_paraReal.dir/build

CMakeFiles/test_paraReal.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_paraReal.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_paraReal.dir/clean

CMakeFiles/test_paraReal.dir/depend:
	cd /home/ubuntu/Desktop/PacsProject/project/tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Desktop/PacsProject/project/tests /home/ubuntu/Desktop/PacsProject/project/tests /home/ubuntu/Desktop/PacsProject/project/tests /home/ubuntu/Desktop/PacsProject/project/tests /home/ubuntu/Desktop/PacsProject/project/tests/CMakeFiles/test_paraReal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_paraReal.dir/depend

