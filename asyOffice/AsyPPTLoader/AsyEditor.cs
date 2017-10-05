using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace AsyPPTLoader
{
    public partial class AsyEditor : Form
    {
        public AsyEditor()
        {
            InitializeComponent();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
        }

        private void btnAccept_Click(object sender, EventArgs e)
        {
            var tmpFile = Path.GetTempFileName();
            StreamWriter sw1 = new StreamWriter(tmpFile);
            sw1.Write(this.textBox1.Text);
            sw1.Close();

            Globals.ThisAddIn.AddAsyCode(tmpFile);

            this.DialogResult = DialogResult.OK;
        }
    }
}
