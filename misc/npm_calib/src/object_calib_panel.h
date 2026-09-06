#ifndef NPM_CALIB_OBJECT_CALIB_PANEL_H
#define NPM_CALIB_OBJECT_CALIB_PANEL_H

#include <string>

#include <rviz/panel.h>

class QLabel;
class QLineEdit;
class QPushButton;
class QVBoxLayout;

namespace npm_calib
{

/** Docked buttons for the object_calib_node Trigger services.
 *
 * The panel owns no calibration state: every button is a std_srvs/Trigger call
 * and the node stays the single writer of the extrinsic. That keeps the panel
 * usable even if the node is restarted mid-session.
 */
class ObjectCalibPanel : public rviz::Panel
{
  Q_OBJECT

public:
  explicit ObjectCalibPanel(QWidget* parent = 0);

  // Persist the namespace field into the .rviz file.
  void save(rviz::Config config) const override;
  void load(const rviz::Config& config) override;

private Q_SLOTS:
  void onSave();
  void onSnap90();
  void onReset();
  void onResetSaved();
  void onZeroTranslation();
  void onZeroRotation();

private:
  /** Call `<ns>/<service>` and report the response in the status label. */
  void callTrigger(const std::string& service);
  void setStatus(const QString& text, bool ok);

  /** Namespace text with the trailing slash stripped and a leading one
   *  guaranteed, so "object_calib" and "/object_calib/" both resolve. */
  std::string resolvedNs() const;

  QPushButton* addButton(QVBoxLayout* layout, const QString& text,
                         const char* slot);

  QLineEdit* ns_edit_;
  QLabel* status_;
};

}  // namespace npm_calib

#endif  // NPM_CALIB_OBJECT_CALIB_PANEL_H
