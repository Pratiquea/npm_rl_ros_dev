#include "object_calib_panel.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include <ros/ros.h>
#include <std_srvs/Trigger.h>

namespace npm_calib
{

namespace
{
const char* kDefaultNs = "/object_calib";
}  // namespace

ObjectCalibPanel::ObjectCalibPanel(QWidget* parent)
  : rviz::Panel(parent), ns_edit_(new QLineEdit(kDefaultNs)), status_(new QLabel)
{
  QVBoxLayout* layout = new QVBoxLayout;

  QHBoxLayout* ns_row = new QHBoxLayout;
  ns_row->addWidget(new QLabel("node"));
  ns_edit_->setToolTip("Namespace of object_calib_node (its private services)");
  ns_row->addWidget(ns_edit_);
  layout->addLayout(ns_row);

  addButton(layout, "Save calibration", SLOT(onSave()));
  addButton(layout, "Snap rotation to 90 deg", SLOT(onSnap90()));
  addButton(layout, "Reset to identity", SLOT(onReset()));
  addButton(layout, "Reset to last saved", SLOT(onResetSaved()));
  addButton(layout, "Zero translation", SLOT(onZeroTranslation()));
  addButton(layout, "Zero rotation", SLOT(onZeroRotation()));

  // The node already formats the pose and the save path; echo that verbatim
  // rather than reformatting it here, so panel and rosout never disagree.
  status_->setWordWrap(true);
  status_->setTextInteractionFlags(Qt::TextSelectableByMouse);
  layout->addWidget(status_);
  layout->addStretch();

  setLayout(layout);
}

QPushButton* ObjectCalibPanel::addButton(QVBoxLayout* layout, const QString& text,
                                         const char* slot)
{
  QPushButton* b = new QPushButton(text);
  connect(b, SIGNAL(clicked()), this, slot);
  layout->addWidget(b);
  return b;
}

std::string ObjectCalibPanel::resolvedNs() const
{
  std::string ns = ns_edit_->text().trimmed().toStdString();
  if (ns.empty())
    ns = kDefaultNs;
  if (ns[0] != '/')
    ns = "/" + ns;
  while (ns.size() > 1 && ns[ns.size() - 1] == '/')
    ns.erase(ns.size() - 1);
  return ns;
}

void ObjectCalibPanel::setStatus(const QString& text, bool ok)
{
  status_->setStyleSheet(ok ? "" : "color: #c0392b;");
  status_->setText(text);
}

void ObjectCalibPanel::callTrigger(const std::string& service)
{
  const std::string name = resolvedNs() + "/" + service;

  // Checked separately so a node that is simply not running reads differently
  // from a service that ran and refused.
  if (!ros::service::exists(name, /*print_failure_reason=*/false))
  {
    setStatus(QString("no such service: %1").arg(QString::fromStdString(name)),
              false);
    return;
  }

  // Synchronous on the Qt thread on purpose: the handlers are a pose update and
  // at worst one YAML write, and a blocking call keeps click -> result ordered.
  std_srvs::Trigger srv;
  if (!ros::service::call(name, srv))
  {
    setStatus(QString("call failed: %1").arg(QString::fromStdString(name)), false);
    return;
  }
  setStatus(QString::fromStdString(srv.response.message), srv.response.success);
}

void ObjectCalibPanel::onSave()            { callTrigger("save"); }
void ObjectCalibPanel::onSnap90()          { callTrigger("snap90"); }
void ObjectCalibPanel::onReset()           { callTrigger("reset"); }
void ObjectCalibPanel::onResetSaved()      { callTrigger("reset_saved"); }
void ObjectCalibPanel::onZeroTranslation() { callTrigger("zero_translation"); }
void ObjectCalibPanel::onZeroRotation()    { callTrigger("zero_rotation"); }

void ObjectCalibPanel::save(rviz::Config config) const
{
  rviz::Panel::save(config);
  config.mapSetValue("Namespace", ns_edit_->text());
}

void ObjectCalibPanel::load(const rviz::Config& config)
{
  rviz::Panel::load(config);
  QString ns;
  if (config.mapGetString("Namespace", &ns))
    ns_edit_->setText(ns);
}

}  // namespace npm_calib

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(npm_calib::ObjectCalibPanel, rviz::Panel)
