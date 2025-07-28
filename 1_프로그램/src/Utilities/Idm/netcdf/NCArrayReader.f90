!> @brief This module contains the NCArrayReaderModule
!!
!! This module defines the netcdf_array_load interface
!! which can read layered (UGRID) and non-layered (STRUCTURED)
!! netcdf arrays stored in modflow6 designated input variables.
!!
!<
module NCArrayReaderModule

  use KindModule, only: DP, I4B, LGP
  use ConstantsModule, only: LINELENGTH
  use SimVariablesModule, only: errmsg
  use SimModule, only: store_error, store_error_filename
  use InputDefinitionModule, only: InputParamDefinitionType
  use ModflowInputModule, only: ModflowInputType
  use NCFileVarsModule, only: NCPackageVarsType
  use SourceCommonModule, only: get_shape_from_string, get_layered_shape
  use NetCDFCommonModule, only: nf_verify
  use netcdf

  implicit none
  private
  public :: netcdf_array_load

  interface netcdf_array_load
    module procedure nc_array_load_int1d, nc_array_load_int2d, &
      nc_array_load_int3d, nc_array_load_dbl1d, nc_array_load_dbl2d, &
      nc_array_load_dbl3d
  end interface netcdf_array_load

contains

  !> @brief does the grid support per layer variables
  !<
  function is_layered(grid) result(layered)
    character(len=*), intent(in) :: grid
    logical(LGP) :: layered
    select case (grid)
    case ('LAYERED MESH')
      layered = .true.
    case ('STRUCTURED')
      layered = .false.
    case default
      layered = .false.
    end select
  end function is_layered

  !> @brief Load NetCDF integer 1D array
  !<
  subroutine nc_array_load_int1d(int1d, mshape, idt, mf6_input, nc_vars, &
                                 input_fname, iout, kper)
    integer(I4B), dimension(:), pointer, contiguous, intent(in) :: int1d
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape !< model shape
    type(InputParamDefinitionType), intent(in) :: idt !< input data type object describing this record
    type(ModflowInputType), intent(in) :: mf6_input
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), intent(in) :: iout
    integer(I4B), optional, intent(in) :: kper
    integer(I4B) :: varid
    logical(LGP) :: layered

    layered = (idt%layered .and. is_layered(nc_vars%grid))

    if (layered) then
      call load_integer1d_layered(int1d, mf6_input, mshape, idt, nc_vars, &
                                  input_fname)
    else
      if (present(kper)) then
        varid = nc_vars%varid(idt%mf6varname, period=kper)
      else
        varid = nc_vars%varid(idt%mf6varname)
      end if
      call load_integer1d_type(int1d, mf6_input, mshape, idt, nc_vars, &
                               varid, input_fname)
    end if
  end subroutine nc_array_load_int1d

  !> @brief Load NetCDF integer 2D array
  !<
  subroutine nc_array_load_int2d(int2d, mshape, idt, mf6_input, nc_vars, &
                                 input_fname, iout)
    integer(I4B), dimension(:, :), pointer, contiguous, intent(in) :: int2d
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape !< model shape
    type(InputParamDefinitionType), intent(in) :: idt !< input data type object describing this record
    type(ModflowInputType), intent(in) :: mf6_input
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), intent(in) :: iout
    integer(I4B) :: varid
    logical(LGP) :: layered

    layered = (idt%layered .and. is_layered(nc_vars%grid))

    if (layered) then
      call load_integer2d_layered(int2d, mf6_input, mshape, idt, nc_vars, &
                                  input_fname)
    else
      varid = nc_vars%varid(idt%mf6varname)
      call load_integer2d_type(int2d, mf6_input, mshape, idt, nc_vars, &
                               varid, input_fname)
    end if
  end subroutine nc_array_load_int2d

  !> @brief Load NetCDF integer 3D array
  !<
  subroutine nc_array_load_int3d(int3d, mshape, idt, mf6_input, nc_vars, &
                                 input_fname, iout)
    integer(I4B), dimension(:, :, :), pointer, contiguous, intent(in) :: int3d
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape !< model shape
    type(InputParamDefinitionType), intent(in) :: idt !< input data type object describing this record
    type(ModflowInputType), intent(in) :: mf6_input
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), intent(in) :: iout
    integer(I4B) :: varid
    logical(LGP) :: layered

    layered = (idt%layered .and. is_layered(nc_vars%grid))

    if (layered) then
      call load_integer3d_layered(int3d, mf6_input, mshape, idt, nc_vars, &
                                  input_fname)
    else
      varid = nc_vars%varid(idt%mf6varname)
      call load_integer3d_type(int3d, mf6_input, mshape, idt, nc_vars, &
                               varid, input_fname)
    end if
  end subroutine nc_array_load_int3d

  !> @brief Load NetCDF double 1D array
  !<
  subroutine nc_array_load_dbl1d(dbl1d, mshape, idt, mf6_input, nc_vars, &
                                 input_fname, iout, kper, iaux)
    real(DP), dimension(:), pointer, contiguous, intent(in) :: dbl1d
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape !< model shape
    type(InputParamDefinitionType), intent(in) :: idt !< input data type object describing this record
    type(ModflowInputType), intent(in) :: mf6_input
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), intent(in) :: iout
    integer(I4B), optional, intent(in) :: kper
    integer(I4B), optional, intent(in) :: iaux
    integer(I4B) :: varid
    logical(LGP) :: layered

    if (present(kper)) then
      layered = (kper > 0 .and. is_layered(nc_vars%grid))
    else
      layered = (idt%layered .and. is_layered(nc_vars%grid))
    end if

    if (layered) then
      if (present(kper)) then
        call load_double1d_layered_spd(dbl1d, mf6_input, mshape, idt, nc_vars, &
                                       kper, input_fname, iaux)
      else
        call load_double1d_layered(dbl1d, mf6_input, mshape, idt, nc_vars, &
                                   input_fname)
      end if
    else
      if (present(kper)) then
        call load_double1d_spd(dbl1d, mf6_input, mshape, idt, nc_vars, &
                               kper, input_fname, iaux)
      else
        varid = nc_vars%varid(idt%mf6varname)
        call load_double1d_type(dbl1d, mf6_input, mshape, idt, nc_vars, &
                                varid, input_fname)
      end if
    end if
  end subroutine nc_array_load_dbl1d

  !> @brief Load NetCDF double 2D array
  !<
  subroutine nc_array_load_dbl2d(dbl2d, mshape, idt, mf6_input, nc_vars, &
                                 input_fname, iout)
    real(DP), dimension(:, :), pointer, contiguous, intent(in) :: dbl2d
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape !< model shape
    type(InputParamDefinitionType), intent(in) :: idt !< input data type object describing this record
    type(ModflowInputType), intent(in) :: mf6_input
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), intent(in) :: iout
    integer(I4B) :: varid
    logical(LGP) :: layered

    layered = (idt%layered .and. is_layered(nc_vars%grid))

    if (layered) then
      call load_double2d_layered(dbl2d, mf6_input, mshape, idt, nc_vars, &
                                 input_fname)
    else
      varid = nc_vars%varid(idt%mf6varname)
      call load_double2d_type(dbl2d, mf6_input, mshape, idt, nc_vars, &
                              varid, input_fname)
    end if
  end subroutine nc_array_load_dbl2d

  !> @brief Load NetCDF double 3D array
  !<
  subroutine nc_array_load_dbl3d(dbl3d, mshape, idt, mf6_input, nc_vars, &
                                 input_fname, iout)
    real(DP), dimension(:, :, :), pointer, contiguous, intent(in) :: dbl3d
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape !< model shape
    type(InputParamDefinitionType), intent(in) :: idt !< input data type object describing this record
    type(ModflowInputType), intent(in) :: mf6_input
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), intent(in) :: iout
    integer(I4B) :: varid
    logical(LGP) :: layered

    layered = (idt%layered .and. is_layered(nc_vars%grid))

    if (layered) then
      call load_double3d_layered(dbl3d, mf6_input, mshape, idt, nc_vars, &
                                 input_fname)
    else
      varid = nc_vars%varid(idt%mf6varname)
      call load_double3d_type(dbl3d, mf6_input, mshape, idt, nc_vars, &
                              varid, input_fname)
    end if
  end subroutine nc_array_load_dbl3d

  !> @brief load type 1d integer
  !<
  subroutine load_integer1d_type(int1d, mf6_input, mshape, idt, nc_vars, &
                                 varid, input_fname)
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: int1d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    integer(I4B), intent(in) :: varid
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), allocatable :: array_shape
    integer(I4B), dimension(:, :, :), contiguous, pointer :: int3d_ptr
    integer(I4B), dimension(:, :), contiguous, pointer :: int2d_ptr
    integer(I4B) :: nvals

    ! initialize
    nvals = 0

    if (idt%shape == 'NODES') then
      ! set number of values
      nvals = product(mshape)
      if (size(mshape) == 3) then
        int3d_ptr(1:mshape(3), 1:mshape(2), 1:mshape(1)) => int1d(1:nvals)
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, int3d_ptr), &
                       nc_vars%nc_fname)
      else if (size(mshape) == 2) then
        int2d_ptr(1:mshape(2), 1:mshape(1)) => int1d(1:nvals)
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, int2d_ptr), &
                       nc_vars%nc_fname)
      else if (size(mshape) == 1) then
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, int1d), nc_vars%nc_fname)
      end if
    else
      ! interpret shape
      call get_shape_from_string(idt%shape, array_shape, mf6_input%mempath)
      ! set nvals
      nvals = array_shape(1)
      ! read and set data
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, int1d), nc_vars%nc_fname)
    end if
  end subroutine load_integer1d_type

  !> @brief load type 1d integer layered
  !<
  subroutine load_integer1d_layered(int1d, mf6_input, mshape, idt, nc_vars, &
                                    input_fname)
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: int1d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), allocatable :: layer_shape
    integer(I4B) :: nlay, varid
    integer(I4B) :: k, ncpl
    integer(I4B) :: index_start, index_stop
    integer(I4B), dimension(:), contiguous, pointer :: int1d_ptr

    nullify (int1d_ptr)

    call get_layered_shape(mshape, nlay, layer_shape)

    ncpl = product(layer_shape)
    index_start = 1
    do k = 1, nlay
      varid = nc_vars%varid(idt%mf6varname, layer=k)
      index_stop = index_start + ncpl - 1
      int1d_ptr(1:ncpl) => int1d(index_start:index_stop)
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, int1d_ptr), &
                     nc_vars%nc_fname)
      index_start = index_stop + 1
    end do
  end subroutine load_integer1d_layered

  !> @brief load type 2d integer
  !<
  subroutine load_integer2d_type(int2d, mf6_input, mshape, idt, nc_vars, varid, &
                                 input_fname)
    integer(I4B), dimension(:, :), contiguous, pointer, intent(in) :: int2d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    integer(I4B), intent(in) :: varid
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), contiguous, pointer :: int1d_ptr
    integer(I4B), dimension(:), allocatable :: array_shape
    integer(I4B) :: ncpl, nlay

    nullify (int1d_ptr)

    if (nc_vars%grid == 'STRUCTURED') then
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, int2d), nc_vars%nc_fname)
    else if (nc_vars%grid == 'LAYERED MESH') then
      call get_layered_shape(mshape, nlay, array_shape)
      ncpl = product(array_shape)
      int1d_ptr(1:ncpl) => int2d(:, :)
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, int1d_ptr), &
                     nc_vars%nc_fname)
    end if
  end subroutine load_integer2d_type

  !> @brief load type 2d integer layered
  !<
  subroutine load_integer2d_layered(int2d, mf6_input, mshape, idt, nc_vars, &
                                    input_fname)
    integer(I4B), dimension(:, :), contiguous, pointer, intent(in) :: int2d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), allocatable :: layer_shape
    integer(I4B) :: k
    integer(I4B) :: ncpl, nlay, varid
    integer(I4B), dimension(:), contiguous, pointer :: int1d_ptr

    nullify (int1d_ptr)

    if (size(mshape) == 3) then
      write (errmsg, '(a,a,a)') &
        'Layered netcdf read not supported for DIS int2d type ('// &
        trim(idt%tagname)//').'
      call store_error(errmsg)
      call store_error_filename(input_fname)
    else if (size(mshape) == 2) then
      call get_layered_shape(mshape, nlay, layer_shape)
      ncpl = layer_shape(1)
      do k = 1, nlay
        varid = nc_vars%varid(idt%mf6varname, layer=k)
        int1d_ptr(1:ncpl) => int2d(1:ncpl, k)
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, int1d_ptr), &
                       nc_vars%nc_fname)
      end do
    end if
  end subroutine load_integer2d_layered

  !> @brief load type 3d integer
  !<
  subroutine load_integer3d_type(int3d, mf6_input, mshape, idt, nc_vars, varid, &
                                 input_fname)
    integer(I4B), dimension(:, :, :), contiguous, pointer, intent(in) :: int3d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    integer(I4B), intent(in) :: varid
    character(len=*), intent(in) :: input_fname
    call nf_verify(nf90_get_var(nc_vars%ncid, varid, int3d), nc_vars%nc_fname)
  end subroutine load_integer3d_type

  !> @brief load type 3d integer layered
  !<
  subroutine load_integer3d_layered(int3d, mf6_input, mshape, idt, nc_vars, &
                                    input_fname)
    integer(I4B), dimension(:, :, :), contiguous, pointer, intent(in) :: int3d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), allocatable :: layer_shape
    integer(I4B) :: k !, i, j
    integer(I4B) :: ncpl, nlay, varid
    integer(I4B) :: index_start, index_stop
    integer(I4B), dimension(:), contiguous, pointer :: int1d_ptr

    nullify (int1d_ptr)
    index_start = 1
    call get_layered_shape(mshape, nlay, layer_shape)
    ncpl = product(layer_shape)

    do k = 1, nlay
      varid = nc_vars%varid(idt%mf6varname, layer=k)
      index_stop = index_start + ncpl - 1
      int1d_ptr(1:ncpl) => int3d(:, :, k:k)
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, int1d_ptr), &
                     nc_vars%nc_fname)
      index_start = index_stop + 1
    end do
  end subroutine load_integer3d_layered

  !> @brief load type 1d double
  !<
  subroutine load_double1d_type(dbl1d, mf6_input, mshape, idt, nc_vars, &
                                varid, input_fname)
    real(DP), dimension(:), contiguous, pointer, intent(in) :: dbl1d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    integer(I4B), intent(in) :: varid
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), allocatable :: array_shape
    real(DP), dimension(:, :, :), contiguous, pointer :: dbl3d_ptr
    real(DP), dimension(:, :), contiguous, pointer :: dbl2d_ptr
    integer(I4B) :: nvals

    ! initialize
    nvals = 0

    if (idt%shape == 'NODES') then
      ! set number of values
      nvals = product(mshape)
      if (size(mshape) == 3) then
        dbl3d_ptr(1:mshape(3), 1:mshape(2), 1:mshape(1)) => dbl1d(1:nvals)
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl3d_ptr), &
                       nc_vars%nc_fname)
      else if (size(mshape) == 2) then
        dbl2d_ptr(1:mshape(2), 1:mshape(1)) => dbl1d(1:nvals)
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl2d_ptr), &
                       nc_vars%nc_fname)
      else if (size(mshape) == 1) then
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl1d), nc_vars%nc_fname)
      end if
    else
      ! interpret shape
      call get_shape_from_string(idt%shape, array_shape, mf6_input%mempath)
      ! set nvals
      nvals = array_shape(1)
      ! read and set data
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl1d), nc_vars%nc_fname)
    end if
  end subroutine load_double1d_type

  !> @brief load type 1d double
  !<
  subroutine load_double1d_spd(dbl1d, mf6_input, mshape, idt, nc_vars, &
                               iper, input_fname, iaux)
    use ConstantsModule, only: DNODATA
    real(DP), dimension(:), contiguous, pointer, intent(in) :: dbl1d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    integer(I4B), intent(in) :: iper
    character(len=*), intent(in) :: input_fname
    integer(I4B), optional, intent(in) :: iaux
    real(DP), dimension(:, :, :), contiguous, pointer :: dbl3d
    integer(I4B) :: nvals, varid
    integer(I4B) :: n, i, j, k

    ! initialize
    nvals = 0

    ! set varid
    if (present(iaux)) then
      varid = nc_vars%varid(idt%mf6varname, period=iper, iaux=iaux)
    else
      varid = nc_vars%varid(idt%mf6varname, period=iper)
    end if

    if (idt%shape == 'NODES') then
      ! TODO future support
      write (errmsg, '(a)') &
        'IDM NetCDF load_double1d_spd NODES var shape not supported => '// &
        trim(idt%tagname)
      call store_error(errmsg)
      call store_error_filename(input_fname)
    else if (idt%shape == 'NCPL' .or. idt%shape == 'NAUX NCPL') then

      if (size(mshape) == 3) then
        allocate (dbl3d(mshape(3), mshape(2), mshape(1)))
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl3d), &
                       nc_vars%nc_fname)
        n = 0
        do k = 1, size(dbl3d, dim=3)
          do i = 1, size(dbl3d, dim=2)
            do j = 1, size(dbl3d, dim=1)
              if (n < size(dbl1d)) then
                n = n + 1
              else
                n = 1
              end if
              if (dbl3d(j, i, k) /= DNODATA) then
                dbl1d(n) = dbl3d(j, i, k)
              end if
            end do
          end do
        end do

      else if (size(mshape) == 2) then
        ! TODO
        write (errmsg, '(a)') &
          'IDM NetCDF load_double1d_spd DISV model not supported => '// &
          trim(idt%tagname)
        call store_error(errmsg)
        call store_error_filename(input_fname)
      end if
    end if
  end subroutine load_double1d_spd

  !> @brief load type 1d double layered
  !<
  subroutine load_double1d_layered(dbl1d, mf6_input, mshape, idt, nc_vars, &
                                   input_fname)
    real(DP), dimension(:), contiguous, pointer, intent(in) :: dbl1d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), allocatable :: layer_shape
    integer(I4B) :: nlay, varid
    integer(I4B) :: k, ncpl
    integer(I4B) :: index_start, index_stop
    real(DP), dimension(:), contiguous, pointer :: dbl1d_ptr

    nullify (dbl1d_ptr)
    index_start = 1
    call get_layered_shape(mshape, nlay, layer_shape)
    ncpl = product(layer_shape)

    do k = 1, nlay
      varid = nc_vars%varid(idt%mf6varname, layer=k)
      index_stop = index_start + ncpl - 1
      dbl1d_ptr(1:ncpl) => dbl1d(index_start:index_stop)
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl1d_ptr), &
                     nc_vars%nc_fname)
      index_start = index_stop + 1
    end do
  end subroutine load_double1d_layered

  !> @brief load type 1d double layered
  !<
  subroutine load_double1d_layered_spd(dbl1d, mf6_input, mshape, idt, nc_vars, &
                                       iper, input_fname, iaux)
    use ConstantsModule, only: DNODATA
    real(DP), dimension(:), contiguous, pointer, intent(in) :: dbl1d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    integer(I4B), intent(in) :: iper
    character(len=*), intent(in) :: input_fname
    integer(I4B), optional, intent(in) :: iaux
    integer(I4B), dimension(:), allocatable :: layer_shape
    integer(I4B) :: nlay, varid
    integer(I4B) :: k, n, ncpl
    integer(I4B) :: index_start, index_stop
    real(DP), dimension(:), contiguous, pointer :: dbl1d_ptr

    call get_layered_shape(mshape, nlay, layer_shape)
    ncpl = product(layer_shape)
    allocate (dbl1d_ptr(ncpl))

    do k = 1, nlay
      index_start = 1
      index_stop = index_start + ncpl - 1
      if (present(iaux)) then
        varid = nc_vars%varid(idt%mf6varname, layer=k, period=iper, iaux=iaux)
      else
        varid = nc_vars%varid(idt%mf6varname, layer=k, period=iper)
      end if
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl1d_ptr), &
                     nc_vars%nc_fname)
      do n = 1, ncpl
        if (dbl1d_ptr(n) /= DNODATA) then
          dbl1d(n) = dbl1d_ptr(n)
        end if
      end do
    end do

    ! cleanup
    deallocate (dbl1d_ptr)
  end subroutine load_double1d_layered_spd

  !> @brief load type 2d double
  !<
  subroutine load_double2d_type(dbl2d, mf6_input, mshape, idt, nc_vars, varid, &
                                input_fname)
    real(DP), dimension(:, :), contiguous, pointer, intent(in) :: dbl2d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    integer(I4B), intent(in) :: varid
    character(len=*), intent(in) :: input_fname
    real(DP), dimension(:), contiguous, pointer :: dbl1d_ptr
    integer(I4B), dimension(:), allocatable :: array_shape
    integer(I4B) :: ncpl, nlay

    nullify (dbl1d_ptr)

    if (nc_vars%grid == 'STRUCTURED') then
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl2d), nc_vars%nc_fname)
    else if (nc_vars%grid == 'LAYERED MESH') then
      call get_layered_shape(mshape, nlay, array_shape)
      ncpl = product(array_shape)
      dbl1d_ptr(1:ncpl) => dbl2d(:, :)
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl1d_ptr), &
                     nc_vars%nc_fname)
    end if
  end subroutine load_double2d_type

  !> @brief load type 2d double layered
  !<
  subroutine load_double2d_layered(dbl2d, mf6_input, mshape, idt, nc_vars, &
                                   input_fname)
    real(DP), dimension(:, :), contiguous, pointer, intent(in) :: dbl2d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), allocatable :: layer_shape
    integer(I4B) :: k
    integer(I4B) :: ncpl, nlay, varid
    real(DP), dimension(:), contiguous, pointer :: dbl1d_ptr

    nullify (dbl1d_ptr)

    if (size(mshape) == 3) then
      write (errmsg, '(a,a,a)') &
        'Layered netcdf read not supported for DIS dbl2d type ('// &
        trim(idt%tagname)//').'
      call store_error(errmsg)
      call store_error_filename(input_fname)
    else if (size(mshape) == 2) then
      call get_layered_shape(mshape, nlay, layer_shape)
      ncpl = layer_shape(1)
      do k = 1, nlay
        varid = nc_vars%varid(idt%mf6varname, layer=k)
        dbl1d_ptr(1:ncpl) => dbl2d(1:ncpl, k)
        call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl1d_ptr), &
                       nc_vars%nc_fname)
      end do
    end if
  end subroutine load_double2d_layered

  !> @brief load type 3d double
  !<
  subroutine load_double3d_type(dbl3d, mf6_input, mshape, idt, nc_vars, varid, &
                                input_fname)
    real(DP), dimension(:, :, :), contiguous, pointer, intent(in) :: dbl3d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    integer(I4B), intent(in) :: varid
    character(len=*), intent(in) :: input_fname
    !
    call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl3d), nc_vars%nc_fname)
  end subroutine load_double3d_type

  !> @brief load type 3d double layered
  !<
  subroutine load_double3d_layered(dbl3d, mf6_input, mshape, idt, nc_vars, &
                                   input_fname)
    real(DP), dimension(:, :, :), contiguous, pointer, intent(in) :: dbl3d
    type(ModflowInputType), intent(in) :: mf6_input
    integer(I4B), dimension(:), contiguous, pointer, intent(in) :: mshape
    type(InputParamDefinitionType), intent(in) :: idt
    type(NCPackageVarsType), pointer, intent(in) :: nc_vars
    character(len=*), intent(in) :: input_fname
    integer(I4B), dimension(:), allocatable :: layer_shape
    integer(I4B) :: k !, i, j
    integer(I4B) :: ncpl, nlay, varid
    integer(I4B) :: index_start, index_stop
    real(DP), dimension(:), contiguous, pointer :: dbl1d_ptr

    nullify (dbl1d_ptr)

    call get_layered_shape(mshape, nlay, layer_shape)

    ncpl = product(layer_shape)
    index_start = 1
    do k = 1, nlay
      varid = nc_vars%varid(idt%mf6varname, layer=k)
      index_stop = index_start + ncpl - 1
      dbl1d_ptr(1:ncpl) => dbl3d(:, :, k:k)
      call nf_verify(nf90_get_var(nc_vars%ncid, varid, dbl1d_ptr), &
                     nc_vars%nc_fname)
      index_start = index_stop + 1
    end do
  end subroutine load_double3d_layered

end module NCArrayReaderModule
