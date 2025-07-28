!> @brief This module contains the Mf6FileGridInputModule
!!
!! This module contains the routines for reading period block
!! array based input.
!!
!<
module Mf6FileGridInputModule

  use KindModule, only: I4B, DP, LGP
  use ConstantsModule, only: DZERO, IZERO, LINELENGTH, LENVARNAME, &
                             LENTIMESERIESNAME, LENAUXNAME
  use SimVariablesModule, only: errmsg
  use SimModule, only: store_error, store_error_filename
  use InputDefinitionModule, only: InputParamDefinitionType
  use MemoryManagerModule, only: mem_allocate, mem_reallocate, mem_setptr
  use CharacterStringModule, only: CharacterStringType
  use BlockParserModule, only: BlockParserType
  use ModflowInputModule, only: ModflowInputType, getModflowInput
  use BoundInputContextModule, only: BoundInputContextType, ReadStateVarType
  use TimeArraySeriesManagerModule, only: TimeArraySeriesManagerType, &
                                          tasmanager_cr
  use AsciiInputLoadTypeModule, only: AsciiDynamicPkgLoadBaseType

  implicit none
  private
  public :: BoundGridInputType

  !> @brief Ascii grid based dynamic loader type
  !<
  type, extends(AsciiDynamicPkgLoadBaseType) :: BoundGridInputType
    integer(I4B) :: tas_active !< Are TAS6 inputs defined
    type(CharacterStringType), dimension(:), contiguous, &
      pointer :: aux_tasnames !< array of AUXVAR TAS names
    type(CharacterStringType), dimension(:), contiguous, &
      pointer :: param_tasnames !< array of dynamic param TAS names
    type(ReadStateVarType), dimension(:), allocatable :: param_reads !< read states for current load
    type(TimeArraySeriesManagerType), pointer :: tasmanager !< TAS manager
    type(BoundInputContextType) :: bound_context
  contains
    procedure :: ainit => bndgrid_init
    procedure :: df => bndgrid_df
    procedure :: ad => bndgrid_ad
    procedure :: rp => bndgrid_rp
    procedure :: destroy => bndgrid_destroy
    procedure :: reset => bndgrid_reset
    procedure :: init_charstr1d
    procedure :: params_alloc => bndgrid_params_alloc
    procedure :: param_load => bndgrid_param_load
    procedure :: tas_arrays_alloc => bndgrid_tas_arrays_alloc
    procedure :: tas_links_create => bndgrid_tas_links_create
  end type BoundGridInputType

contains

  subroutine bndgrid_init(this, mf6_input, component_name, &
                          component_input_name, input_name, &
                          iperblock, parser, iout)
    use MemoryManagerModule, only: get_isize
    use BlockParserModule, only: BlockParserType
    use LoadMf6FileModule, only: LoadMf6FileType
    class(BoundGridInputType), intent(inout) :: this
    type(ModflowInputType), intent(in) :: mf6_input
    character(len=*), intent(in) :: component_name
    character(len=*), intent(in) :: component_input_name
    character(len=*), intent(in) :: input_name
    integer(I4B), intent(in) :: iperblock
    type(BlockParserType), pointer, intent(inout) :: parser
    integer(I4B), intent(in) :: iout
    type(LoadMf6FileType) :: loader
    type(CharacterStringType), dimension(:), pointer, &
      contiguous :: tas_fnames
    character(len=LINELENGTH) :: fname
    integer(I4B) :: tas6_size, n

    ! initialize base type
    call this%DynamicPkgLoadType%init(mf6_input, component_name, &
                                      component_input_name, &
                                      input_name, iperblock, iout)
    ! initialize
    nullify (this%aux_tasnames)
    nullify (this%param_tasnames)
    this%tas_active = 0
    this%iout = iout

    ! load static input
    call loader%load(parser, mf6_input, this%nc_vars, this%input_name, iout)

    ! create tasmanager
    allocate (this%tasmanager)
    call tasmanager_cr(this%tasmanager, modelname=this%mf6_input%component_name, &
                       iout=this%iout)

    ! determine if TAS6 files were provided in OPTIONS block
    call get_isize('TAS6_FILENAME', this%mf6_input%mempath, tas6_size)
    if (tas6_size > 0) then
      this%tas_active = 1
      call mem_setptr(tas_fnames, 'TAS6_FILENAME', this%mf6_input%mempath)
      ! add files to tasmanager
      do n = 1, size(tas_fnames)
        fname = tas_fnames(n)
        call this%tasmanager%add_tasfile(fname)
      end do
    end if

    ! initialize input context memory
    call this%bound_context%create(mf6_input, this%readasarrays)

    ! allocate dfn params
    call this%params_alloc()

    ! allocate memory for storing TAS strings
    call this%tas_arrays_alloc()
  end subroutine bndgrid_init

  subroutine bndgrid_df(this)
    class(BoundGridInputType), intent(inout) :: this !< Mf6FileGridInputType
    call this%tasmanager%tasmanager_df()
  end subroutine bndgrid_df

  subroutine bndgrid_ad(this)
    class(BoundGridInputType), intent(inout) :: this !< Mf6FileGridInputType
    call this%tasmanager%ad()
  end subroutine bndgrid_ad

  subroutine bndgrid_rp(this, parser)
    use MemoryManagerModule, only: mem_setptr
    use BlockParserModule, only: BlockParserType
    use InputDefinitionModule, only: InputParamDefinitionType
    use DefinitionSelectModule, only: get_param_definition_type
    use ArrayHandlersModule, only: ifind
    use SourceCommonModule, only: ifind_charstr
    use IdmLoggerModule, only: idm_log_header, idm_log_close, idm_log_var
    class(BoundGridInputType), intent(inout) :: this !< Mf6FileGridInputType
    type(BlockParserType), pointer, intent(inout) :: parser
    logical(LGP) :: endOfBlock, netcdf
    character(len=LINELENGTH) :: keyword, param_tag
    type(InputParamDefinitionType), pointer :: idt
    integer(I4B) :: iaux, iparam
    character(len=LENTIMESERIESNAME) :: tas_name
    integer(I4B), dimension(:), pointer, contiguous :: int1d

    ! reset for this period
    call this%reset()

    ! log lst file header
    call idm_log_header(this%mf6_input%component_name, &
                        this%mf6_input%subcomponent_name, this%iout)

    ! read array block
    do
      ! initialize
      iaux = 0
      netcdf = .false.

      ! read next line
      call parser%GetNextLine(endOfBlock)
      if (endOfBlock) exit
      ! read param_tag
      call parser%GetStringCaps(param_tag)

      ! is param tag an auxvar?
      iaux = ifind_charstr(this%bound_context%auxname_cst, param_tag)
      ! any auvxar corresponds to the definition tag 'AUX'
      if (iaux > 0) param_tag = 'AUX'

      ! set input definition
      idt => get_param_definition_type(this%mf6_input%param_dfns, &
                                       this%mf6_input%component_type, &
                                       this%mf6_input%subcomponent_type, &
                                       'PERIOD', param_tag, this%input_name)
      ! look for TAS and NetCDF keywords
      call parser%GetStringCaps(keyword)
      if (keyword == 'TIMEARRAYSERIES') then
        if (this%tas_active /= 0) then
          call parser%GetStringCaps(tas_name)
          if (param_tag == 'AUX') then
            this%aux_tasnames(iaux) = tas_name
          else
            iparam = ifind(this%param_names, param_tag)
            this%param_tasnames(iparam) = tas_name
            this%param_reads(iparam)%invar = 2
          end if
          ! log variable
          call idm_log_var(param_tag, this%mf6_input%mempath, this%iout, .true.)
          ! cycle to next input param
          cycle
        else
          ! TODO: throw error
        end if
      else if (keyword == 'NETCDF') then
        netcdf = .true.
      end if

      ! read and load the parameter
      call this%param_load(parser, idt, this%mf6_input%mempath, netcdf, iaux)
    end do

    ! check if layer index variable was read
    ! TODO: assumes layer index variable is always in scope
    if (this%param_reads(1)%invar == 0) then
      ! set to default of 1 without updating invar
      idt => get_param_definition_type(this%mf6_input%param_dfns, &
                                       this%mf6_input%component_type, &
                                       this%mf6_input%subcomponent_type, &
                                       'PERIOD', this%param_names(1), &
                                       this%input_name)
      call mem_setptr(int1d, idt%mf6varname, this%mf6_input%mempath)
      int1d = 1
    end if

    if (this%tas_active /= 0) then
      call this%tas_links_create(parser%iuactive)
    end if

    ! log lst file header
    call idm_log_close(this%mf6_input%component_name, &
                       this%mf6_input%subcomponent_name, this%iout)
  end subroutine bndgrid_rp

  subroutine bndgrid_destroy(this)
    class(BoundGridInputType), intent(inout) :: this !< Mf6FileGridInputType
    !
    ! deallocate tasmanager
    call this%tasmanager%da()
    deallocate (this%tasmanager)
    nullify (this%tasmanager)
  end subroutine bndgrid_destroy

  subroutine bndgrid_reset(this)
    class(BoundGridInputType), intent(inout) :: this !< BoundGridInputType
    integer(I4B) :: n, m

    if (this%tas_active /= 0) then
      ! reset tasmanager
      call this%tasmanager%reset(this%mf6_input%subcomponent_name)
      ! reinitialize tas name arrays
      call this%init_charstr1d('AUXTASNAME', this%input_name)
      call this%init_charstr1d('PARAMTASNAME', this%input_name)
    end if

    do n = 1, this%nparam
      ! reset read state
      this%param_reads(n)%invar = 0
    end do

    ! explicitly reset auxvar array each period
    do m = 1, this%bound_context%ncpl
      do n = 1, this%bound_context%naux
        this%bound_context%auxvar(n, m) = DZERO
      end do
    end do
  end subroutine bndgrid_reset

  subroutine init_charstr1d(this, varname, input_name)
    use MemoryManagerModule, only: mem_setptr
    class(BoundGridInputType) :: this
    character(len=*), intent(in) :: varname
    character(len=*), intent(in) :: input_name
    type(CharacterStringType), dimension(:), pointer, &
      contiguous :: charstr1d
    integer(I4B) :: n
    call mem_setptr(charstr1d, varname, this%mf6_input%mempath)
    do n = 1, size(charstr1d)
      charstr1d(n) = ''
    end do
  end subroutine init_charstr1d

  subroutine bndgrid_params_alloc(this)
    class(BoundGridInputType), intent(inout) :: this !< BoundGridInputType
    character(len=LENVARNAME) :: rs_varname
    integer(I4B), pointer :: intvar
    integer(I4B) :: iparam

    ! set in scope param names
    call this%bound_context%bound_params(this%param_names, this%nparam, &
                                         this%input_name)
    call this%bound_context%allocate_arrays()

    ! allocate and set param_reads pointer array
    allocate (this%param_reads(this%nparam))

    ! store read state variable pointers
    do iparam = 1, this%nparam
      ! allocate and store name of read state variable
      rs_varname = this%bound_context%rsv_alloc(this%param_names(iparam))
      call mem_setptr(intvar, rs_varname, this%mf6_input%mempath)
      this%param_reads(iparam)%invar => intvar
      this%param_reads(iparam)%invar = 0
    end do
  end subroutine bndgrid_params_alloc

  subroutine bndgrid_param_load(this, parser, idt, mempath, netcdf, iaux)
    use TdisModule, only: kper
    use MemoryManagerModule, only: mem_setptr
    use ArrayHandlersModule, only: ifind
    use InputDefinitionModule, only: InputParamDefinitionType
    use DefinitionSelectModule, only: get_param_definition_type
    use Double1dReaderModule, only: read_dbl1d
    use Double2dReaderModule, only: read_dbl2d
    use Integer1dReaderModule, only: read_int1d
    use LoadNCInputModule, only: netcdf_read_array
    use IdmLoggerModule, only: idm_log_var
    class(BoundGridInputType), intent(inout) :: this !< BoundGridInputType
    type(BlockParserType), intent(in) :: parser
    type(InputParamDefinitionType), intent(in) :: idt
    character(len=*), intent(in) :: mempath
    logical(LGP), intent(in) :: netcdf
    integer(I4B), intent(in) :: iaux
    integer(I4B), dimension(:), pointer, contiguous :: int1d
    real(DP), dimension(:), pointer, contiguous :: dbl1d
    real(DP), dimension(:, :), pointer, contiguous :: dbl2d
    integer(I4B) :: iparam, n

    select case (idt%datatype)
    case ('INTEGER1D')
      call mem_setptr(int1d, idt%mf6varname, mempath)
      if (netcdf) then
        call netcdf_read_array(int1d, this%bound_context%mshape, idt, &
                               this%mf6_input, this%nc_vars, this%input_name, &
                               this%iout, kper)
      else
        call read_int1d(parser, int1d, idt%mf6varname)
      end if
      call idm_log_var(int1d, idt%tagname, mempath, this%iout)
    case ('DOUBLE1D')
      call mem_setptr(dbl1d, idt%mf6varname, mempath)
      if (netcdf) then
        call netcdf_read_array(dbl1d, this%bound_context%mshape, idt, &
                               this%mf6_input, this%nc_vars, this%input_name, &
                               this%iout, kper)
      else
        call read_dbl1d(parser, dbl1d, idt%mf6varname)
      end if
      call idm_log_var(dbl1d, idt%tagname, mempath, this%iout)
    case ('DOUBLE2D')
      call mem_setptr(dbl2d, idt%mf6varname, mempath)
      allocate (dbl1d(this%bound_context%ncpl))
      if (netcdf) then
        call netcdf_read_array(dbl1d, this%bound_context%mshape, idt, &
                               this%mf6_input, this%nc_vars, this%input_name, &
                               this%iout, kper, iaux)
      else
        call read_dbl1d(parser, dbl1d, idt%mf6varname)
      end if
      do n = 1, this%bound_context%ncpl
        dbl2d(iaux, n) = dbl1d(n)
      end do
      call idm_log_var(dbl1d, idt%tagname, mempath, this%iout)
      deallocate (dbl1d)
    case default
      errmsg = 'IDM unimplemented. Mf6FileGridInput::param_load &
               &datatype='//trim(idt%datatype)
      call store_error(errmsg)
      call store_error_filename(this%input_name)
    end select

    ! if param is tracked set read state
    iparam = ifind(this%param_names, idt%tagname)
    if (iparam > 0) then
      this%param_reads(iparam)%invar = 1
    end if
  end subroutine bndgrid_param_load

  subroutine bndgrid_tas_arrays_alloc(this)
    use MemoryManagerModule, only: mem_allocate
    class(BoundGridInputType), intent(inout) :: this !< BoundGridInputType

    ! count params other than AUX
    if (this%tas_active /= 0) then
      call mem_allocate(this%aux_tasnames, LENTIMESERIESNAME, &
                        this%bound_context%naux, 'AUXTASNAME', &
                        this%mf6_input%mempath)
      call mem_allocate(this%param_tasnames, LENTIMESERIESNAME, this%nparam, &
                        'PARAMTASNAME', this%mf6_input%mempath)
      call this%init_charstr1d('AUXTASNAME', this%input_name)
      call this%init_charstr1d('PARAMTASNAME', this%input_name)
    else
      call mem_allocate(this%aux_tasnames, LENTIMESERIESNAME, 0, &
                        'AUXTASNAME', this%mf6_input%mempath)
      call mem_allocate(this%param_tasnames, LENTIMESERIESNAME, 0, &
                        'PARAMTASNAME', this%mf6_input%mempath)
    end if
  end subroutine bndgrid_tas_arrays_alloc

  ! FLUX and SFAC are handled in model context
  subroutine bndgrid_tas_links_create(this, inunit)
    use InputDefinitionModule, only: InputParamDefinitionType
    use DefinitionSelectModule, only: get_param_definition_type
    class(BoundGridInputType), intent(inout) :: this !< BoundGridInputType
    integer(I4B), intent(in) :: inunit
    type(InputParamDefinitionType), pointer :: idt
    ! non-contiguous because a slice of bound is passed
    real(DP), dimension(:), pointer :: auxArrayPtr, bndArrayPtr
    real(DP), dimension(:), pointer, contiguous :: bound
    integer(I4B), dimension(:), pointer, contiguous :: nodelist
    character(len=LENTIMESERIESNAME) :: tas_name
    character(len=LENAUXNAME) :: aux_name
    logical :: convertFlux
    integer(I4B) :: n

    ! initialize
    nullify (auxArrayPtr)
    nullify (bndArrayPtr)
    nullify (nodelist)
    convertflux = .false.

    ! Create AUX Time Array Series links
    do n = 1, this%bound_context%naux
      tas_name = this%aux_tasnames(n)
      if (tas_name /= '') then
        ! set auxvar pointer
        auxArrayPtr => this%bound_context%auxvar(n, :)
        aux_name = this%bound_context%auxname_cst(n)
        call this%tasmanager%MakeTasLink(this%mf6_input%subcomponent_name, &
                                         auxArrayPtr, this%bound_context%iprpak, &
                                         tas_name, aux_name, convertFlux, &
                                         nodelist, inunit)
      end if
    end do

    ! Create BND Time Array Series links
    do n = 1, this%nparam
      ! assign param definition pointer
      idt => get_param_definition_type(this%mf6_input%param_dfns, &
                                       this%mf6_input%component_type, &
                                       this%mf6_input%subcomponent_type, &
                                       'PERIOD', this%param_names(n), &
                                       this%input_name)
      if (idt%timeseries) then
        if (this%param_reads(n)%invar == 2) then
          tas_name = this%param_tasnames(n)
          call mem_setptr(bound, idt%mf6varname, this%mf6_input%mempath)
          ! set bound pointer
          bndArrayPtr => bound(:)
          call this%tasmanager%MakeTasLink(this%mf6_input%subcomponent_name, &
                                           bndArrayPtr, &
                                           this%bound_context%iprpak, &
                                           tas_name, idt%mf6varname, &
                                           convertFlux, nodelist, inunit)
        end if
      end if
    end do
  end subroutine bndgrid_tas_links_create

end module Mf6FileGridInputModule
