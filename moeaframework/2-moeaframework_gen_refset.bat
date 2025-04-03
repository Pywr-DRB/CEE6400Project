@echo off
setlocal enabledelayedexpansion

echo Form reference set

:: Default epsilon value if no argument is given
if "%~1"=="" (
    set "epsilon=0.01,0.01,0.01"
    set "pause_at_end=True"
) else (
    set "epsilon=%~1"
    set "pause_at_end=False"
)

:: Check if Java is installed and callable
:: Please download and install the Java Development Kit (JDK) from:
:: https://www.oracle.com/java/technologies/downloads/#jdk23-windows

:: Check the expected JAR file
set "jarFile="
set "jarURL=https://github.com/MOEAFramework/MOEAFramework/releases/download/v4.5/MOEAFramework-4.5-Demo.jar"
set "jarName=MOEAFramework-4.5-Demo.jar"

for %%F in (*Demo.jar) do set "jarFile=%%F"

if not defined jarFile (
    echo.
    echo [ERROR] MOEAFramework Demo JAR file not found in the current directory.
    echo The required file can be downloaded from:
    echo %jarURL%
    echo.

    echo Downloading %jarName%...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%jarURL%', '%jarName%')"

    if exist "%jarName%" (
        echo Download complete.
        set "jarFile=%jarName%"
    ) else (
        echo [ERROR] Failed to download the file.
        pause
        exit /b 1
    )

)

:: Loop over all .set files in the current directory
for %%F in (*.set) do (
    echo Processing %%F

    java -cp "%jarFile%" ^
        org.moeaframework.analysis.tools.ReferenceSetMerger ^
        --output borg.ref ^
        --epsilon %epsilon% ^
        "%%F"
)

echo Reference set complete.
if /I "%pause_at_end%"=="True" pause