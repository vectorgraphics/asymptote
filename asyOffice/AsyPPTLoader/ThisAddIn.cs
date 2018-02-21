using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using PowerPoint = Microsoft.Office.Interop.PowerPoint;
using Office = Microsoft.Office.Core;
using System.Diagnostics;
using System.IO;

namespace AsyPPTLoader
{
    public partial class ThisAddIn
    {
        public void AddAsyCode(string filePath)
        {
            string tmpPath = Path.GetTempFileName() + Guid.NewGuid().ToString() + ".svg";
            string asyExecArgs = string.Format("-f svg -nobatchView -o \"{0}\" \"{1}\"", tmpPath, filePath);
            var asyProcess = new Process();

            string asyProc = "asy";
            if (Properties.Settings.Default.useCustomRuntime)
            {
                asyProc = Properties.Settings.Default.customAsyRuntime;
            }

            var procInfo = new ProcessStartInfo(asyProc, asyExecArgs)
            {
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardError = true
            };

            asyProcess.StartInfo = procInfo;

            asyProcess.Start();
            asyProcess.WaitForExit();

            if (File.Exists(tmpPath))
            {
                PowerPoint.Slide activeSlide = Application.ActiveWindow.View.Slide;
                PowerPoint.Shape asyPic = activeSlide.Shapes.AddPicture(tmpPath, Office.MsoTriState.msoFalse, Office.MsoTriState.msoTrue, 0, 0);
                
            }
            else
            {
                string errStr = asyProcess.StandardError.ReadToEnd();
                System.Windows.Forms.MessageBox.Show("Asymptote Error. Error:\n", "Error",
                    System.Windows.Forms.MessageBoxButtons.OK, System.Windows.Forms.MessageBoxIcon.Error);
            }
        }

        private void ThisAddIn_Startup(object sender, System.EventArgs e)
        {
        }

        private void ThisAddIn_Shutdown(object sender, System.EventArgs e)
        {
        }

        #region VSTO generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InternalStartup()
        {
            this.Startup += new System.EventHandler(ThisAddIn_Startup);
            this.Shutdown += new System.EventHandler(ThisAddIn_Shutdown);
        }
        
        #endregion
    }
}
