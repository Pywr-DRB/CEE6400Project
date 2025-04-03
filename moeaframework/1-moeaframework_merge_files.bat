@echo off
setlocal enabledelayedexpansion

echo Extract set from runtime file

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

:: Count the number of elements in the epsilon array to set dimension
set dimension=0
for %%A in (%epsilon%) do set /A dimension+=1

:: Loop over all .runtime files in the current directory
for %%F in (*.runtime) do (
    set "input_file=%%F"
    set "output_file=%%~nF.set"

    echo Processing !input_file!

    java -cp "%jarFile%" ^
        org.moeaframework.analysis.tools.ResultFileMerger ^
        --dimension %dimension% ^
        --output "!output_file!" ^
        --epsilon "%epsilon%" ^
        "!input_file!"
)

echo Merging complete.

if /I "%pause_at_end%"=="True" pause
