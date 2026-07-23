; Inno Setup script for Music Mastery Enhancer.
;
; Packages the PyInstaller-built MusicMasteryEnhancer.exe (see
; installer/music_mastery_enhancer.spec) into a guided Windows installer.
;
; Model weights (BS-RoFormer, resemble-enhance) are deliberately NOT bundled here —
; per the "download_first_run" architecture decision, they are fetched by the
; app's own first-run SetupWizard (app/ui/setup_wizard.py) on first launch instead,
; keeping this installer small.
;
; Build with (from repo root, after the PyInstaller build has produced dist\MusicMasteryEnhancer.exe):
;     iscc installer\setup_script.iss
; See installer/README.md for the full build procedure.

#define MyAppName "Music Mastery Enhancer"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Music Mastery Enhancer"
#define MyAppExeName "MusicMasteryEnhancer.exe"

[Setup]
AppId={{B6C9E4D2-6F1A-4B7E-9C3D-2A8F5E1D7C90}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=MusicMasteryEnhancer-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
; Model weights are downloaded by the app on first run, not bundled here, so the
; installer stays small; no extra disk-space checks are required for them.
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Only the single frozen exe produced by PyInstaller is packaged. Do NOT add
; model weight files (*.ckpt, *.pth) here — those are downloaded on first run.
Source: "..\dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Music Mastery Enhancer"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Music Mastery Enhancer"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
