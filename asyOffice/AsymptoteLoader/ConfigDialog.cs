using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AsymptoteLoader
{
    public partial class ConfigDialog : Form
    {
        public ConfigDialog()
        {
            InitializeComponent();
        }

        private void btnSelectAsy_Click(object sender, EventArgs e)
        {
            OpenFileDialog op1 = new OpenFileDialog();
            op1.DefaultExt = "exe";
            op1.Filter = "asy.exe|asy.exe";
            var result = op1.ShowDialog();
            if (result == DialogResult.OK)
            {
                this.txtCustomAsy.Text = op1.FileName;
            }
        }

        private void btnSpecifyRuntime_CheckedChanged(object sender, EventArgs e)
        {
            this.txtCustomAsy.ReadOnly = !btnSpecifyRuntime.Checked;
        }

        private void ConfigDialog_Load(object sender, EventArgs e)
        {
            this.btnSpecifyRuntime.Checked = Properties.Settings.Default.useCustomRuntime;
            if (Properties.Settings.Default.useCustomRuntime)
            {
                this.txtCustomAsy.ReadOnly = false;
                this.txtCustomAsy.Text = Properties.Settings.Default.customAsyRuntime;
            }
            else
            {
                this.txtCustomAsy.ReadOnly = true;
            }
        }

        private void saveSettings()
        {
            Properties.Settings.Default.useCustomRuntime = this.btnSpecifyRuntime.Checked;
            Properties.Settings.Default.customAsyRuntime = this.txtCustomAsy.Text;
            Properties.Settings.Default.Save();
        }

        private void btnAccept_Click(object sender, EventArgs e)
        {
            saveSettings();
            this.Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.Close();
        }
    }
}
