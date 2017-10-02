using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Office.Tools.Ribbon;
using Word = Microsoft.Office.Interop.Word;
using Office = Microsoft.Office.Core;
using System.Windows.Forms;

namespace AsymptoteLoader
{
    public partial class AsyLoaderRibbon
    {
        private void AsyLoaderRibbon_Load(object sender, RibbonUIEventArgs e)
        {

        }

        private void btnAddAsy_Click(object sender, RibbonControlEventArgs e)
        {
            AsyEditor editor1 = new AsyEditor();
            editor1.ShowDialog();
            editor1.Dispose();
        }

        private void btnLoadAsy_Click(object sender, RibbonControlEventArgs e)
        {
            OpenFileDialog op1 = new OpenFileDialog();
            op1.DefaultExt = "asy";
            op1.Filter = "Asymptote Code (*.asy)|*.asy";
            var result = op1.ShowDialog();
            if (result == DialogResult.OK)
            {
                string fileName = op1.FileName;
                Globals.ThisAddIn.AddAsyCode(fileName);
            }
        }

        private void btnConfigure_Click(object sender, RibbonControlEventArgs e)
        {
            ConfigDialog configDiag1 = new ConfigDialog();
            configDiag1.ShowDialog();
            configDiag1.Dispose();
        }
    }
}
