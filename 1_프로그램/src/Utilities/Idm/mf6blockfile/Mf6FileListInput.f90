!> @brief This module contains the Mf6FileListInputModule
!!
!! This module contains the routines for reading period block
!! list based input.
!!
!<
module Mf6FileListInputModule

  use KindModule, only: I4B, DP, LGP
  use ConstantsModule, only: LINELENGTH, LENBOUNDNAME
  use InputDefinitionModule, only: InputParamDefinitionType
  use MemoryManagerModule, only: mem_setptr
  use CharacterStringModule, only: CharacterStringType
  use ModflowInputModule, only: ModflowInputType, getModflowInput
  use TimeSeriesManagerModule, only: TimeSeriesManagerType, tsmanager_cr
  use StructArrayModule, only: StructArrayType, constructStructArray, &
                               destructStructArray
  use AsciiInputLoadTypeModule, only: AsciiDynamicPkgLoadBaseType
  use BoundInputContextModule, only: BoundInputContextType

  implicit none
  private
  public :: BoundListInputType

  !> @brief Boundary package list loader.
  !!
  !! Creates boundary input context for a package,
  !! (e.g. CHD or MAW) and updates that context in
  !! read and prepare (RP) routines.
  !!
  !<
  type, extends(AsciiDynamicPkgLoadBaseType) :: BoundListInputType
    type(TimeSeriesManagerType), pointer :: tsmanager => null()
    type(StructArrayType), pointer :: structarray => null()
    type(BoundInputContextType) :: bound_context
    integer(I4B) :: ts_active
    integer(I4B) :: iboundname
  contains
    procedure :: ainit => bndlist_init
    procedure :: df => bndlist_df
    procedure :: ad => bndlist_ad
    procedure :: reset => bndlist_reset
    procedure :: rp => bndlist_rp
    procedure :: destroy => bndlist_destroy
    procedure :: ts_link_bnd => bndlist_ts_link_bnd
    procedure :: ts_link_aux => bndlist_ts_link_aux
    procedure :: ts_link => bndlist_ts_link
    procedure :: ts_update => bndlist_ts_update
    procedure :: create_structarray => bndlist_create_structarray
  end type BoundListInputType

contains

  subroutine bndlist_init(this, mf6_input, component_name, component_input_name, &
                          input_name, iperblock, parser, iout)
    use InputOutputModule, only: getunit
    use MemoryManagerModule, only: get_isize
    use BlockParserModule, only: BlockParserType
    use LoadMf6FileModule, only: LoadMf6FileType
    class(BoundListInputType), intent(inout) :: this
    type(ModflowInputType), intent(in) :: mf6_input
    character(len=*), intent(in) :: component_name
    character(len=*), intent(in) :: component_input_name
    character(len=*), intent(in) :: input_name
    integer(I4B), intent(in) :: iperblock
    type(BlockParserType), pointer, intent(inout) :: parser
    integer(I4B), intent(in) :: iout
    type(LoadMf6FileType) :: loader
    type(CharacterStringType), dimension(:), pointer, &
      contiguous :: ts_fnames
    character(len=LINELENGTH) :: fname
    integer(I4B) :: ts6_size, n

    ! init loader
    call this%DynamicPkgLoadType%init(mf6_input, component_name, &
                                      component_input_name, input_name, &
                                      iperblock, iout)
    ! initialize scalars
    this%iboundname = 0
    this%ts_active = 0

    ! load static input
    call loader%load(parser, mf6_input, this%nc_vars, this%input_name, iout)

    ! create tsmanager
    allocate (this%tsmanager)
    call tsmanager_cr(this%tsmanager, iout)

    ! determine if TS6 files were provided in OPTIONS block
    call get_isize('TS6_FILENAME', this%mf6_input%mempath, ts6_size)
    if (ts6_size > 0) then
      this%ts_active = 1
      call mem_setptr(ts_fnames, 'TS6_FILENAME', this%mf6_input%mempath)
      do n = 1, size(ts_fnames)
        fname = ts_fnames(n)
        call this%tsmanager%add_tsfile(fname, GetUnit())
      end do
    end if

    ! initialize package input context
    call this%bound_context%create(mf6_input, this%readasarrays)

    ! store in scope SA cols for list input
    call this%bound_context%bound_params(this%param_names, this%nparam, &
                                         this%input_name, create=.false.)
    ! construct and set up the struct array object
    call this%create_structarray()

    ! finalize input context setup
    call this%bound_context%allocate_arrays()
  end subroutine bndlist_init

  subroutine bndlist_df(this)
    class(BoundListInputType), intent(inout) :: this !< ListInputType
    ! define tsmanager
    call this%tsmanager%tsmanager_df()
  end subroutine bndlist_df

  subroutine bndlist_ad(this)
    class(BoundListInputType), intent(inout) :: this !< ListInputType
    ! advance timeseries
    call this%tsmanager%ad()
  end subroutine bndlist_ad

  subroutine bndlist_reset(this)
    class(BoundListInputType), intent(inout) :: this !< ListInputType
    ! reset tsmanager
    call this%tsmanager%reset(this%mf6_input%subcomponent_name)
  end subroutine bndlist_reset

  subroutine bndlist_rp(this, parser)
    use BlockParserModule, only: BlockParserType
    use LoadMf6FileModule, only: read_control_record
    use StructVectorModule, only: StructVectorType
    use IdmLoggerModule, only: idm_log_header, idm_log_close
    class(BoundListInputType), intent(inout) :: this
    type(BlockParserType), pointer, intent(inout) :: parser
    integer(I4B) :: ibinary
    integer(I4B) :: oc_inunit
    logical(LGP) :: ts_active

    call this%reset()
    ibinary = read_control_record(parser, oc_inunit, this%iout)

    ! log lst file header
    call idm_log_header(this%mf6_input%component_name, &
                        this%mf6_input%subcomponent_name, this%iout)

    if (ibinary == 1) then
      this%bound_context%nbound = &
        this%structarray%read_from_binary(oc_inunit, this%iout)
      call parser%terminateblock()
      close (oc_inunit)
    else
      ts_active = (this%ts_active /= 0)
      this%bound_context%nbound = &
        this%structarray%read_from_parser(parser, ts_active, this%iout)
    end if

    ! update ts links
    if (this%ts_active /= 0) then
      call this%ts_update(this%structarray)
    end if

    ! close logging statement
    call idm_log_close(this%mf6_input%component_name, &
                       this%mf6_input%subcomponent_name, this%iout)
  end subroutine bndlist_rp

  subroutine bndlist_destroy(this)
    class(BoundListInputType), intent(inout) :: this !< BoundListInputType
    !
    ! deallocate tsmanager
    call this%tsmanager%da()
    deallocate (this%tsmanager)
    nullify (this%tsmanager)
    !
    ! deallocate StructArray
    call destructStructArray(this%structarray)
    call this%bound_context%destroy()
  end subroutine bndlist_destroy

  subroutine bndlist_ts_link_bnd(this, structvector, ts_strloc)
    use TimeSeriesLinkModule, only: TimeSeriesLinkType
    use TimeSeriesManagerModule, only: read_value_or_time_series
    use StructVectorModule, only: StructVectorType, TSStringLocType
    class(BoundListInputType), intent(inout) :: this
    type(StructVectorType), pointer, intent(in) :: structvector
    type(TSStringLocType), pointer, intent(in) :: ts_strloc
    real(DP), pointer :: bndElem
    type(TimeSeriesLinkType), pointer :: tsLinkBnd
    type(StructVectorType), pointer :: sv_bound
    character(len=LENBOUNDNAME) :: boundname

    nullify (tsLinkBnd)

    ! set bound element
    bndElem => structvector%dbl1d(ts_strloc%row)

    ! set link
    call read_value_or_time_series(ts_strloc%token, ts_strloc%row, &
                                   ts_strloc%structarray_col, bndElem, &
                                   this%mf6_input%subcomponent_name, &
                                   'BND', this%tsmanager, &
                                   this%bound_context%iprpak, tsLinkBnd)
    if (associated(tsLinkBnd)) then
      ! set variable name
      tsLinkBnd%Text = structvector%idt%mf6varname
      ! set boundname if provided
      if (this%bound_context%inamedbound > 0) then
        sv_bound => this%structarray%get(this%iboundname)
        boundname = sv_bound%charstr1d(ts_strloc%row)
        tsLinkBnd%BndName = boundname
      end if
    end if
  end subroutine bndlist_ts_link_bnd

  subroutine bndlist_ts_link_aux(this, structvector, ts_strloc)
    use TimeSeriesLinkModule, only: TimeSeriesLinkType
    use TimeSeriesManagerModule, only: read_value_or_time_series
    use StructVectorModule, only: StructVectorType, TSStringLocType
    class(BoundListInputType), intent(inout) :: this
    type(StructVectorType), pointer, intent(in) :: structvector
    type(TSStringLocType), pointer, intent(in) :: ts_strloc
    real(DP), pointer :: bndElem
    type(TimeSeriesLinkType), pointer :: tsLinkAux
    type(StructVectorType), pointer :: sv_bound
    character(len=LENBOUNDNAME) :: boundname

    nullify (tsLinkAux)

    ! set bound element
    bndElem => structvector%dbl2d(ts_strloc%col, ts_strloc%row)

    ! set link
    call read_value_or_time_series(ts_strloc%token, ts_strloc%row, &
                                   ts_strloc%structarray_col, bndElem, &
                                   this%mf6_input%subcomponent_name, &
                                   'AUX', this%tsmanager, &
                                   this%bound_context%iprpak, tsLinkAux)
    if (associated(tsLinkAux)) then
      ! set variable name
      tsLinkAux%Text = this%bound_context%auxname_cst(ts_strloc%col)
      ! set boundname if provided
      if (this%bound_context%inamedbound > 0) then
        sv_bound => this%structarray%get(this%iboundname)
        boundname = sv_bound%charstr1d(ts_strloc%row)
        tsLinkAux%BndName = boundname
      end if
    end if
  end subroutine bndlist_ts_link_aux

  subroutine bndlist_ts_update(this, structarray)
    use SimModule, only: count_errors, store_error_filename
    use StructVectorModule, only: TSStringLocType
    use StructVectorModule, only: StructVectorType
    class(BoundListInputType), intent(inout) :: this
    type(StructArrayType), pointer, intent(inout) :: structarray
    integer(I4B) :: n, m
    type(TSStringLocType), pointer :: ts_strloc
    type(StructVectorType), pointer :: sv

    do m = 1, structarray%count()
      sv => structarray%get(m)
      if (sv%idt%timeseries) then
        do n = 1, sv%ts_strlocs%count()
          ts_strloc => sv%get_ts_strloc(n)
          call this%ts_link(sv, ts_strloc)
        end do
        call sv%clear()
      end if
    end do

    ! terminate if errors were detected
    if (count_errors() > 0) then
      call store_error_filename(this%input_name)
    end if
  end subroutine bndlist_ts_update

  subroutine bndlist_ts_link(this, structvector, ts_strloc)
    use StructVectorModule, only: StructVectorType, TSStringLocType
    class(BoundListInputType), intent(inout) :: this
    type(StructVectorType), pointer, intent(in) :: structvector
    type(TSStringLocType), pointer, intent(in) :: ts_strloc
    select case (structvector%memtype)
    case (2) ! dbl1d
      call this%ts_link_bnd(structvector, ts_strloc)
    case (6) ! dbl2d
      call this%ts_link_aux(structvector, ts_strloc)
    case default
    end select
  end subroutine bndlist_ts_link

  subroutine bndlist_create_structarray(this)
    use InputDefinitionModule, only: InputParamDefinitionType
    use DefinitionSelectModule, only: get_param_definition_type
    class(BoundListInputType), intent(inout) :: this
    type(InputParamDefinitionType), pointer :: idt
    integer(I4B) :: icol

    ! construct and set up the struct array object
    this%structarray => constructStructArray(this%mf6_input, this%nparam, &
                                             this%bound_context%maxbound, 0, &
                                             this%mf6_input%mempath, &
                                             this%mf6_input%component_mempath)
    ! set up struct array
    do icol = 1, this%nparam
      idt => get_param_definition_type(this%mf6_input%param_dfns, &
                                       this%mf6_input%component_type, &
                                       this%mf6_input%subcomponent_type, &
                                       'PERIOD', &
                                       this%param_names(icol), this%input_name)
      ! allocate variable in memory manager
      call this%structarray%mem_create_vector(icol, idt)
      ! store boundname index when found
      if (idt%mf6varname == 'BOUNDNAME') this%iboundname = icol
    end do
  end subroutine bndlist_create_structarray

end module Mf6FileListInputModule
