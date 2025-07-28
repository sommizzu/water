submodule(SfrModule) SfrModuleConstant
contains

  !> @brief Method for solving for simple reaches
    !!
    !! Method to solve the continuity equation for a SFR package
    !! reach using the simple approach.
    !!
  !<
  module procedure sfr_calc_constant
  real(DP) :: qsrc

  this%stage(n) = this%sstage(n)
  d1 = max(DZERO, this%stage(n) - this%strtop(n))

  call this%sfr_calc_qsource(n, d1, qsrc)

  if (this%sfr_gwf_conn(n) == 1) then
    call this%sfr_calc_qgwf(n, d1, hgwf, qgwf)
    qgwf = -qgwf
  else
    qgwf = DZERO
  end if

  ! leakage exceeds inflow
  if (qgwf > qsrc) then
    d1 = DZERO
    call this%sfr_calc_qsource(n, d1, qsrc)
    qgwf = qsrc
  end if

  ! set qd
  qd = qsrc - qgwf

  end procedure sfr_calc_constant

end submodule
