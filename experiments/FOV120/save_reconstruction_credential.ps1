$ErrorActionPreference = 'Stop'
$credentialPath = Join-Path $env:USERPROFILE '.ssh/fov120_paracloud.credential.xml'
$securePassword = Read-Host 'Password for scxi717@BSCC-N56R5 (input is hidden)' -AsSecureString
if ($securePassword.Length -eq 0) { throw 'Empty password; nothing saved.' }
$credential = [System.Management.Automation.PSCredential]::new('scxi717@BSCC-N56R5', $securePassword)
# Export-Clixml encrypts SecureString with Windows DPAPI for this user/machine.
$credential | Export-Clixml -LiteralPath $credentialPath
$credentialAcl = [System.Security.AccessControl.FileSecurity]::new()
$credentialUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$credentialAcl.SetOwner($credentialUser)
$credentialAcl.SetAccessRuleProtection($true, $false)
$credentialAcl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($credentialUser, 'FullControl', 'Allow'))
Set-Acl -LiteralPath $credentialPath -AclObject $credentialAcl
$securePassword.Dispose()
Write-Host 'Encrypted credential saved for this Windows user. No password was printed.'
