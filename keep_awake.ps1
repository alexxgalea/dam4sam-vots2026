$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$t = Add-Type -MemberDefinition $sig -Name SP -Namespace W -PassThru
$ES_CONTINUOUS       = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED  = [uint32]"0x00000001"
$ES_AWAYMODE_REQ     = [uint32]"0x00000040"
[void]$t::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQ)
Write-Output "keep-awake asserted (pid=$PID)"
# Block until killed
while ($true) { Start-Sleep -Seconds 600 }
