// ======================================================
// XIAO ESP32S3 (SENSE)
// ON-DEVICE BLE SENSOR FUSION - CENTRAL/CONTROLLER + ML - v14
// (on-device-ble-sensor-fusion-v14)
//
// WHAT'S NEW IN v14:
//   0. [v14.1] "FETCH ESP32 MODEL TO BROWSER" - ROUND 2. Real-world
//      testing showed v14's single-retry fix still stalled partway
//      through the ~131-chunk model transfer (consistently around the
//      same chunk count each time - a BLE indicate() TX-queue-full
//      symptom, not a bad link). myServerSendBinary() now retries a
//      stuck notify() in a bounded loop (myNotifyWithRetry(), up
//      to 3s per chunk) instead of giving up after one 30ms retry -
//      enough patience for the queue to actually drain. The browser's
//      own stale-transfer watchdog (index-v14.html) was bumped from
//      4s to 8s to match, so it doesn't cut off a legitimate recovery
//      in progress.
//   1. CONSISTENT RESULT OUTPUT - Infer and Quick Infer used to print
//      two DIFFERENT Serial formats ("Predicted: X (...)" vs "top: X =
//      Y%, (...)", the second missing per-class labels entirely), and
//      neither matched the Nano33's own short "On-device inference: X
//      conf=Y%" line. All three now go through one shared
//      myPrintResultLine(), giving the exact same style everywhere,
//      matching what index-v14.html shows on screen:
//        "On-device result: 1normal (0unknown=3%, 1normal=90%, 2issue=7%,)"
//   2. "FETCH ESP32 MODEL TO BROWSER" ACTUALLY COMPLETES NOW -
//      myServerSendBinary()'s v13 fix (a blind delay(15) between
//      indicate() calls) reduced but didn't eliminate drops on the
//      ~131-chunk model transfer - a delay alone doesn't confirm
//      anything actually arrived. v14 checks indicate()'s own return
//      value for every chunk (same "with response" principle already
//      used successfully for myPushModelToNano33()'s WRITE direction),
//      retries once on a false, and aborts with a clear Serial message
//      instead of silently sending a truncated package the browser
//      then waits on forever.
//   3. HEARTBEAT / BUSY-TRANSFER INTERACTION - see the Nano33 sketch's
//      v14 header for the full reasoning (same logic applies here):
//      traced through and confirmed CAPTURE/QUICK/INFER/QUICKINFER
//      transfers can't currently be interleaved by a heartbeat notify,
//      since both directions run to completion inside one loop() call.
//      No functional change needed here as a result, just documenting
//      it since it was asked about directly.
//   4. Forward declarations extended to cover the new/changed
//      functions below (myPrintResultLine, myServerSendResult).
//
// Role: SELECTABLE. Same feature set as v05/v06/v07 (CENTRAL to the
// Nano 33 BLE Sense peripheral - connects, requests fused sensor
// windows on demand, stores to SD by class, trains a Conv1D+Dense
// classifier on-device, runs live inference; PLUS PERIPHERAL to your
// webpage (index-v05.html) so a browser can connect to it directly)
// EXCEPT which of those two roles actually starts at boot is a
// menu-selectable, NVS-persisted setting instead of always-both.
//
// WHAT'S NEW IN v08 (fixes "ESP32 doesn't show up in the browser's
// connection window at all, even in Peripheral/Dual BLE mode"):
//   1. ADVERTISING PAYLOAD OVERFLOW FIXED - myStartBLEPeripheralServer()
//      was advertising BOTH the 128-bit MY_BLE_SERVICE_UUID (18 bytes)
//      AND the board's name (17 bytes) in one legacy advertising
//      packet, totalling ~38 bytes against a hard 31-byte limit.
//      NimBLEAdvertising::start() was failing SILENTLY (return value
//      was never checked) - the board never actually went on air, so
//      it could never appear in navigator.bluetooth.requestDevice()'s
//      chooser, in ANY mode. Fixed by dropping the service UUID from
//      the advertising packet (the webpage filters by namePrefix, not
//      advertised services, so this was never needed for connecting -
//      GATT service discovery still finds MY_BLE_SERVICE_UUID fine
//      once the browser is connected). adv->start()'s return value is
//      now checked and a clear error is printed if it ever fails again.
//
// WHAT WAS NEW IN v07 (fixes for the v05/v06 "crashes a few times at
// startup" + timing-related browser-can't-see-ESP32 reports):
//   1. BLE MODE TOGGLE - Serial menu command 'b' cycles through four
//      modes (DUAL / PERIPHERAL ONLY / CENTRAL ONLY / OFF), saved to
//      NVS via Preferences so it survives reboot without recompiling.
//      'z' restarts immediately to apply a newly-selected mode. The
//      current mode is shown in the menu, the 's' status line, and the
//      idle OLED screen. Running single-role (esp. PERIPHERAL ONLY)
//      sidesteps the single-radio central-scan-vs-advertise contention
//      that caused the intermittent boot crashes and made the browser
//      unable to find the board while it was mid-scan.
//   2. BOOT ORDER FIXED - even in DUAL mode, myStartBLEPeripheralServer()
//      (advertising) now runs BEFORE the blocking 5s myBLEConnectToNano33()
//      scan, not after. Previously the ESP32 wasn't advertising at all
//      for several seconds after boot (scan + connect + calibration),
//      which is exactly the window where the browser's device picker
//      would come up empty. Now the browser can find/connect within
//      ~1s of power-on regardless of whether the Nano33 link is up yet.
//   3. Central-only actions ('r' reconnect, both from Serial and from
//      the browser's "r" command) are now no-ops with a clear message
//      when the active mode doesn't include the central role, instead
//      of silently attempting a scan that can never succeed.
//   4. Everything else (model push/pull, browser command dispatch,
//      training, inference) is unchanged from v05/v06 - see those
//      headers below for the original feature history.
//
// WHAT WAS NEW IN v05 (kept for reference):
//   1. DUAL BLE ROLE - the ESP32S3 is central to the Nano33 (unchanged)
//      AND peripheral to a browser (new). The Serial menu still works
//      exactly as before; browser control is additive, not a replacement.
//   2. MODEL PUSH/PULL - two new characteristics (UUIDs ...af20 family):
//        MODEL_CHAR  (write, chunked)  - a browser-trained model (see
//                     index-v05.html's Train panel) can be pushed
//                     IN to this board over BLE and loaded live.
//        RESULT_CHAR (notify) - this board's own Infer/Quick-Infer
//                     results are now also broadcast to the browser,
//                     not just printed to Serial/OLED.
//      This board can also forward whatever model it currently holds
//      (its own trained weights, OR one just received from the
//      browser) onward to the Nano33 over the existing central link,
//      via myPushModelToNano33() - browser command "PUSHTONANO".
//   3. TRAINING STAYS ESP32-ONLY ON THIS BOARD as before (myActionTrain,
//      unchanged) - the browser can ALSO train now (mirrored JS
//      architecture in index-v05.html), but the Nano33 NEVER trains,
//      per your instructions - it only ever receives a finished model
//      and runs forward-pass inference (see nano33-v05.ino).
//   4. Browser commands accepted on the new peripheral control char
//      mirror the Serial menu 1:1: "CAPTURE", "QUICK", "TRAIN",
//      "INFER", "QUICKINFER", "PUSHTONANO", plus 'r'/'k'/'s'/'d' single
//      letters. Handled asynchronously in loop() (see PART 2.5 below) -
//      never directly inside a BLE callback, to keep the NimBLE stack
//      happy with these longer blocking operations (Train especially).
//
// SENSOR FUSION MODEL INPUT (see the Nano33 sketch header for the
// exact wire format):
//   FAST (Conv1D input): 40 timesteps x 10 channels
//     accelXYZ, gyroXYZ, magXYZ, micLevel
//   STATIC (concatenated in after the conv/pool flatten, once per
//   window - these sensors change far slower than the IMU):
//     pressure, temperature, humidity, proximity, colorR, colorG, colorB
//   Network: Conv1D(k=5,f=8) -> MaxPool/2 -> flatten(144) ++ static(7)
//            = 151 -> Dense(32) -> Dense(16) -> Output(3, softmax)
//
// SD card stores: fused windows in class folders (.dat, raw float32)
// SD card stores: weights in binary (.bin) and .h text header
// Serial monitor and OLED output
//
// A NOTE ON UNTESTED SURFACE AREA: the dual-role NimBLEServer +
// NimBLEClient code below is new in v05 and I could not compile/flash
// it in this environment - the exact NimBLE-Arduino API for
// simultaneous central+peripheral has shifted slightly between library
// major versions (1.4.x vs 2.x). If `createServer()`/`NIMBLE_PROPERTY::*`
// don't match your installed version, check the NimBLE-Arduino
// changelog for that release - the concepts (one NimBLEServer +
// advertising, running alongside the existing NimBLEClient) carry over
// even if a call name moves.
//
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique),
//   Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
// LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// For platformio you need the U8g2 library declared in the platformio.ini file
// lib_deps = olikraus/U8g2 @ ^2.35.30
//            h2zero/NimBLE-Arduino @ ^2.x
// board_build.arduino.memory_type = qio_opi
//
// Arduino IDE: install "NimBLE-Arduino" by h2zero via Library Manager.
// ======================================================


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 0: CORE SYSTEM                                                     ██
// ██████████████████████████████████████████████████████████████████████████████

// [NOTE v09] index-v09.html's new "Save myWeights.h" button generates a
// myFusionWeights.h compatible with this block (same myModel_* names/
// format as myExportHeader() below) - it also adds unused-here
// myModel_calibMean/myModel_calibStd arrays (those are only consumed by
// the Nano33's own USE_BAKED_WEIGHTS block) and an unused-here
// MY_BAKED_NUM_CLASSES #define; both are harmless, ignore them here.
//#define USE_BAKED_WEIGHTS
#ifdef USE_BAKED_WEIGHTS
  #include "myFusionWeights.h"
#endif

#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>
#include <Wire.h>
#include <NimBLEDevice.h>
#include <Preferences.h>   // [NEW v07] persist the BLE mode selection across reboots

U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// FORWARD DECLARATIONS [NEW v07] - kept explicit rather than relying
// on Arduino's auto-prototype generator, per hard-won experience that
// it can silently skip/mis-order prototypes in a sketch this size.
// ======================================================
void myStartBLEPeripheralServer();
bool myBLEConnectToNano33();
void myCalibrate();
void myPrintMenu();
const char* myBleModeName(int m);
void myLoadBleModeFromPrefs();
void mySaveBleModeToPrefs();
void myCycleBleMode();
bool myBleCentralEnabled();
bool myBlePeripheralEnabled();
void myPrintResultLine(int pred, float* probs, String* labels, int numClasses);   // [NEW v14]
bool myNotifyWithRetry(NimBLECharacteristic* ch);   // [CHANGED v14.2] was myIndicateWithRetry
void myExportModelPackage();   // [NEW v14.2] now called from myExportModelAsHex() above its definition too
void myExportModelAsHex();   // [NEW v14.2]
void myServerSendResult(int pred, float* probs, int numClasses);
void myServerSendBinary(const uint8_t* raw, uint32_t totalBytes);
void myActionInferOnce();
void myActionQuickInferOnce();

// ======================================================
// BLE MODE SELECTION [NEW v07]
// Which BLE role(s) this board starts at boot is now a menu-selectable
// setting instead of always running both roles. This exists because
// running central (scanning for + connecting to the Nano33) and
// peripheral (advertising to/serving the browser) at the same time
// puts both jobs on the SAME physical radio - NimBLE-Arduino supports
// it, but the scan/advertise contention around boot and around every
// 'r' reconnect is what produced the intermittent startup crashes and
// the "browser can't find the ESP32" symptom (the board simply wasn't
// advertising yet while it was off scanning for the Nano33). Picking
// PERIPHERAL_ONLY or CENTRAL_ONLY here removes that contention
// entirely; DUAL keeps the old v05/v06 behavior but with the boot
// order fixed (see setup() in PART 8) so advertising starts first.
// Saved to NVS via Preferences, so it survives power-cycling without
// a recompile/reflash - change it with 'b' from the Serial menu.
// ======================================================
enum MyBleMode { BLE_MODE_DUAL = 0, BLE_MODE_PERIPHERAL_ONLY = 1, BLE_MODE_CENTRAL_ONLY = 2, BLE_MODE_OFF = 3 };
int myBleMode = BLE_MODE_DUAL;
Preferences myBlePrefs;

const char* myBleModeName(int m) {
  switch (m) {
    case BLE_MODE_DUAL:            return "DUAL (Nano33 + Browser)";
    case BLE_MODE_PERIPHERAL_ONLY: return "PERIPHERAL ONLY (Browser)";
    case BLE_MODE_CENTRAL_ONLY:    return "CENTRAL ONLY (Nano33)";
    case BLE_MODE_OFF:             return "OFF (no BLE radio)";
    default:                       return "?";
  }
}

void myLoadBleModeFromPrefs() {
  myBlePrefs.begin("fusion", true);   // read-only
  myBleMode = myBlePrefs.getInt("bleMode", BLE_MODE_DUAL);
  myBlePrefs.end();
  if (myBleMode < BLE_MODE_DUAL || myBleMode > BLE_MODE_OFF) myBleMode = BLE_MODE_DUAL;
}

void mySaveBleModeToPrefs() {
  myBlePrefs.begin("fusion", false);
  myBlePrefs.putInt("bleMode", myBleMode);
  myBlePrefs.end();
}

void myCycleBleMode() {
  myBleMode = (myBleMode + 1) % 4;
  mySaveBleModeToPrefs();
  Serial.printf("BLE mode set to: %s\n", myBleModeName(myBleMode));
  Serial.println("Saved - takes effect on next boot. Press 'z' to restart now, or power-cycle later.");
}

bool myBleCentralEnabled()    { return myBleMode == BLE_MODE_DUAL || myBleMode == BLE_MODE_CENTRAL_ONLY; }
bool myBlePeripheralEnabled() { return myBleMode == BLE_MODE_DUAL || myBleMode == BLE_MODE_PERIPHERAL_ONLY; }

// ======================================================
// ML / TASK CONFIGURATION
// ======================================================
#define NUM_CLASSES 3
String myClassLabels[NUM_CLASSES] = {"0unknown", "1normal", "2issue"};

const int myTotalItems = NUM_CLASSES + 2;   // classes + Train + Infer

float LEARNING_RATE  = 0.001f;
int   BATCH_SIZE     = 6;
int   TARGET_EPOCHS  = 30;
int   VALIDATION_SAMPLES = 3;   // last N samples per class held out for validation (0 = disabled)

// ======================================================
// WINDOW / NETWORK ARCHITECTURE
// (must match nano33-sensor-fusion-ble-peripheral-v01.ino's packet layout)
// ======================================================
#define WINDOW_TIMESTEPS   40
#define FAST_CHANNELS      10     // ax,ay,az,gx,gy,gz,mx,my,mz,mic
#define STATIC_FEATURES     7     // pressure,temp,humid,prox,r,g,b

#define FAST_SIZE     (WINDOW_TIMESTEPS * FAST_CHANNELS)   // 400
#define PACKET_FLOATS (FAST_SIZE + STATIC_FEATURES)        // 407
#define PACKET_BYTES  (PACKET_FLOATS * 4)                  // 1628

#define CONV1_KERNEL    5
#define CONV1_FILTERS   8
#define CONV1_OUT_STEPS (WINDOW_TIMESTEPS - CONV1_KERNEL + 1)   // 36
#define POOL1_STEPS     (CONV1_OUT_STEPS / 2)                   // 18
#define CONV1_FLAT      (POOL1_STEPS * CONV1_FILTERS)           // 144
#define COMBINED_SIZE   (CONV1_FLAT + STATIC_FEATURES)          // 151

#define CONV1_WEIGHTS   (CONV1_KERNEL * FAST_CHANNELS * CONV1_FILTERS)  // 400

#define DENSE1_SIZE   32
#define DENSE2_SIZE   16

#define DENSE1_WEIGHTS  (COMBINED_SIZE * DENSE1_SIZE)   // 151 x 32 = 4832
#define DENSE2_WEIGHTS  (DENSE1_SIZE  * DENSE2_SIZE)    //  32 x 16 = 512
#define OUTPUT_WEIGHTS  (DENSE2_SIZE  * NUM_CLASSES)    //  16 x 3  = 48

// ======================================================
// NORMALIZATION
// Fast channels: mean/std learned from ONE calibration window at
// boot (device/machine stationary) - quick, not as thorough as the
// 80-sample calibration in motion-anomaly-v010.ino, but a full
// BLE-round-trip calibration of that length would take ~80 seconds.
// Good enough for v01; re-run calibration (serial 'k') any time.
// Static features: fixed baseline/scale constants below - simple
// z-score-ish normalization, deliberately NOT learned since a single
// window can't estimate a meaningful std for slow-changing sensors.
// TUNE-ME: adjust these baselines/scales for your environment.
// ======================================================
float myFastMean[FAST_CHANNELS] = {0,0,1.0f, 0,0,0, 0,0,0, 0};
float myFastStd [FAST_CHANNELS] = {1,1,1, 1,1,1, 1,1,1, 1};

#define MY_PRESSURE_BASELINE   101.0f
#define MY_PRESSURE_SCALE        5.0f
#define MY_TEMP_BASELINE        20.0f
#define MY_TEMP_SCALE           10.0f
#define MY_HUMIDITY_BASELINE     50.0f
#define MY_HUMIDITY_SCALE       25.0f
#define MY_PROX_SCALE           255.0f
#define MY_COLOR_SCALE          255.0f

// ======================================================
// BLE CENTRAL CONFIGURATION - must match the Nano33 peripheral sketch
// ======================================================
// on-device-ble-sensor-fusion-v04
// v04 adds: 'd' debug toggle (quiet by default now that the BLE link
// is proven working), and 'q' continuous quick-inference mode - pulls
// one instant reading from the Nano33 (QUICK command, ~68 bytes, no
// 1s sampling wait) and replicates it across all 40 Conv1D timesteps
// instead of waiting for a real temporal window. Trades temporal detail
// for speed - good for live monitoring; 'i' (full-window infer) is
// still there when you want a real 1-second capture's worth of motion.
#define MY_NANO33_NAME_PREFIX     "Nano33-Fusion"
#define MY_ESP32_PERIPHERAL_NAME  "ESP32-Fusion-01"   // [NEW v05] name the browser sees when scanning
#define MY_BLE_SERVICE_UUID       "7e400001-b2c3-5d4e-af60-9b3c7d8eaf20"
#define MY_BLE_CONTROL_CHAR_UUID  "7e400002-b2c3-5d4e-af60-9b3c7d8eaf20"
#define MY_BLE_HEARTBEAT_CHAR_UUID "7e400003-b2c3-5d4e-af60-9b3c7d8eaf20"
#define MY_BLE_BINARY_CHAR_UUID   "7e400004-b2c3-5d4e-af60-9b3c7d8eaf20"
#define MY_BLE_MODEL_CHAR_UUID    "7e400005-b2c3-5d4e-af60-9b3c7d8eaf20"  // [NEW v05] write, chunked, model push
#define MY_BLE_RESULT_CHAR_UUID   "7e400006-b2c3-5d4e-af60-9b3c7d8eaf20"  // [NEW v05] notify, inference result

#define MY_CAPTURE_TIMEOUT_MS   6000   // full window: ~1s sample + acked BLE indicate transfer overhead
#define MY_QUICK_TIMEOUT_MS     1500   // quick read: single instant sample, no 1s wait

// ======================================================
// MODEL PACKAGE LAYOUT [NEW v05] - float32LE, must match nano33-v05.ino
// and index-v02.html's myExportModelPackage() bit-for-bit:
//   [0]         numClasses (as a float, e.g. 3.0f)
//   conv1_w(400), conv1_b(8), dense1_w(4832), dense1_b(32),
//   dense2_w(512), dense2_b(16), output_w(48), output_b(3),
//   calibMean(10), calibStd(10)
// On THIS board NUM_CLASSES is a fixed compile-time #define (3), so
// OUTPUT_WEIGHTS is already sized correctly above - the Nano33 is the
// one that has to support a range because it has no compile-time
// knowledge of your class count.
// ======================================================
#define MY_MODEL_PACKAGE_FLOATS (1 + CONV1_WEIGHTS + CONV1_FILTERS + DENSE1_WEIGHTS + DENSE1_SIZE + \
                                  DENSE2_WEIGHTS + DENSE2_SIZE + OUTPUT_WEIGHTS + NUM_CLASSES + \
                                  FAST_CHANNELS + FAST_CHANNELS)
#define MY_MODEL_PACKAGE_BYTES  (MY_MODEL_PACKAGE_FLOATS * 4)   // 23488 bytes at NUM_CLASSES=3

// Debug verbosity - OFF by default now that the link is proven stable.
// Toggle with 'd'. Gates the periodic [status] line, per-chunk transfer
// progress, and one-shot "first packet received" notices - all useful
// while debugging the BLE link, all noise once it's working and you
// just want clean inference output.
bool myDebugVerbose = false;
uint8_t myModelPackageBuf[MY_MODEL_PACKAGE_BYTES];

// ======================================================
// SYSTEM STATE
// ======================================================
bool mySDavailable    = false;
bool myWeightsTrained  = false;
bool myCoreMemoryReady = false;
int  myAdamStep        = 0;

enum MyMode { MODE_MENU, MODE_COLLECT, MODE_INFER, MODE_QUICK_INFER };
MyMode myMode = MODE_MENU;
int myCollectClassIdx = 0;

struct TrainingItem { String path; int label; };
std::vector<TrainingItem> myTrainingData;

// ======================================================
// PSRAM ML BUFFERS
// ======================================================
float* myConv1_w = nullptr;   float* myConv1_b = nullptr;
float* myDense1_w = nullptr;  float* myDense1_b = nullptr;
float* myDense2_w = nullptr;  float* myDense2_b = nullptr;
float* myOutput_w = nullptr;  float* myOutput_b = nullptr;

float* myConv1_w_grad = nullptr;  float* myConv1_b_grad = nullptr;
float* myDense1_w_grad = nullptr; float* myDense1_b_grad = nullptr;
float* myDense2_w_grad = nullptr; float* myDense2_b_grad = nullptr;
float* myOutput_w_grad = nullptr; float* myOutput_b_grad = nullptr;

float* myConv1_w_m = nullptr; float* myConv1_w_v = nullptr;
float* myConv1_b_m = nullptr; float* myConv1_b_v = nullptr;
float* myDense1_w_m = nullptr; float* myDense1_w_v = nullptr;
float* myDense1_b_m = nullptr; float* myDense1_b_v = nullptr;
float* myDense2_w_m = nullptr; float* myDense2_w_v = nullptr;
float* myDense2_b_m = nullptr; float* myDense2_b_v = nullptr;
float* myOutput_w_m = nullptr; float* myOutput_w_v = nullptr;
float* myOutput_b_m = nullptr; float* myOutput_b_v = nullptr;

float* myConv1_output  = nullptr;   // CONV1_OUT_STEPS x CONV1_FILTERS
float* myPool1_output  = nullptr;   // CONV1_FLAT
float* myCombined      = nullptr;   // COMBINED_SIZE (pool1 ++ static)
float* myDense1_output = nullptr;
float* myDense2_output = nullptr;
float* myFinal_output  = nullptr;   // NUM_CLASSES

float* myOutput_delta  = nullptr;
float* myDense2_delta  = nullptr;
float* myDense1_delta  = nullptr;
float* myPool1_delta   = nullptr;
float* myConv1_delta   = nullptr;

// ======================================================
// BLE PACKET RECEIVE BUFFER (raw, unnormalized - one fused window)
// ======================================================
uint8_t myPacketBytes[PACKET_BYTES];
float*  myPacketFloats = (float*)myPacketBytes;

#define QUICK_PACKET_FLOATS  (FAST_CHANNELS + STATIC_FEATURES)  // 17
#define QUICK_PACKET_BYTES   (QUICK_PACKET_FLOATS * 4)          // 68
uint8_t myQuickBytes[QUICK_PACKET_BYTES];
float*  myQuickFloats = (float*)myQuickBytes;

// ======================================================
// UTILITY
// ======================================================
inline float myClip(float v, float mn = -100, float mx = 100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}
inline float myLeakyRelu(float x)      { return x > 0 ? x : 0.1f * x; }
inline float myLeakyReluDeriv(float x) { return x > 0 ? 1.0f : 0.1f; }

void mySoftmax(float* x, int size) {
  float maxVal = x[0];
  for (int i = 1; i < size; i++) if (x[i] > maxVal) maxVal = x[i];
  float sum = 0;
  for (int i = 0; i < size; i++) { x[i] = exp(x[i] - maxVal); sum += x[i]; }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

// Normalize a raw PACKET_FLOATS buffer in place: fast channels via
// learned mean/std, static features via fixed baseline/scale.
void myNormalizePacket(float* buf) {
  for (int t = 0; t < WINDOW_TIMESTEPS; t++) {
    for (int c = 0; c < FAST_CHANNELS; c++) {
      int idx = t * FAST_CHANNELS + c;
      buf[idx] = (buf[idx] - myFastMean[c]) / (myFastStd[c] + 1e-8f);
      buf[idx] = myClip(buf[idx], -5.0f, 5.0f);
    }
  }
  buf[FAST_SIZE + 0] = myClip((buf[FAST_SIZE + 0] - MY_PRESSURE_BASELINE) / MY_PRESSURE_SCALE, -5, 5);
  buf[FAST_SIZE + 1] = myClip((buf[FAST_SIZE + 1] - MY_TEMP_BASELINE)     / MY_TEMP_SCALE,     -5, 5);
  buf[FAST_SIZE + 2] = myClip((buf[FAST_SIZE + 2] - MY_HUMIDITY_BASELINE) / MY_HUMIDITY_SCALE, -5, 5);
  buf[FAST_SIZE + 3] = myClip(buf[FAST_SIZE + 3] / MY_PROX_SCALE,  0, 5);
  buf[FAST_SIZE + 4] = myClip(buf[FAST_SIZE + 4] / MY_COLOR_SCALE, 0, 5);
  buf[FAST_SIZE + 5] = myClip(buf[FAST_SIZE + 5] / MY_COLOR_SCALE, 0, 5);
  buf[FAST_SIZE + 6] = myClip(buf[FAST_SIZE + 6] / MY_COLOR_SCALE, 0, 5);
}

// ======================================================
// MEMORY ALLOCATION
// ======================================================
bool myCheckAlloc(void* ptr, const char* name, size_t bytes) {
  if (ptr) return true;
  Serial.printf("FATAL ALLOC FAIL: %s (%u bytes) - free PSRAM=%d, free heap=%d\n",
                name, (unsigned)bytes, ESP.getFreePsram(), ESP.getFreeHeap());
  return false;
}

bool myAllocateCoreMemory() {
  if (myCoreMemoryReady) return true;
  Serial.println("\n=== Allocating Core Memory (weights + forward buffers) ===");
  Serial.printf("Free PSRAM before allocation: %d bytes\n", ESP.getFreePsram());

  bool ok = true;
  #define MY_ALLOC_CORE(ptr, size, name) \
    ptr = (float*)ps_malloc((size) * sizeof(float)); \
    ok &= myCheckAlloc(ptr, name, (size) * sizeof(float));

  MY_ALLOC_CORE(myConv1_w, CONV1_WEIGHTS, "myConv1_w");
  MY_ALLOC_CORE(myConv1_b, CONV1_FILTERS, "myConv1_b");
  MY_ALLOC_CORE(myDense1_w, DENSE1_WEIGHTS, "myDense1_w");
  MY_ALLOC_CORE(myDense1_b, DENSE1_SIZE, "myDense1_b");
  MY_ALLOC_CORE(myDense2_w, DENSE2_WEIGHTS, "myDense2_w");
  MY_ALLOC_CORE(myDense2_b, DENSE2_SIZE, "myDense2_b");
  MY_ALLOC_CORE(myOutput_w, OUTPUT_WEIGHTS, "myOutput_w");
  MY_ALLOC_CORE(myOutput_b, NUM_CLASSES, "myOutput_b");

  MY_ALLOC_CORE(myConv1_output, CONV1_OUT_STEPS * CONV1_FILTERS, "myConv1_output");
  MY_ALLOC_CORE(myPool1_output, CONV1_FLAT, "myPool1_output");
  MY_ALLOC_CORE(myCombined, COMBINED_SIZE, "myCombined");
  MY_ALLOC_CORE(myDense1_output, DENSE1_SIZE, "myDense1_output");
  MY_ALLOC_CORE(myDense2_output, DENSE2_SIZE, "myDense2_output");
  MY_ALLOC_CORE(myFinal_output, NUM_CLASSES, "myFinal_output");
  #undef MY_ALLOC_CORE

  if (!ok) return false;
  Serial.printf("Free PSRAM after core allocation: %d bytes\n", ESP.getFreePsram());

  float c1std = sqrt(2.0f / (CONV1_KERNEL * FAST_CHANNELS));
  for (int i = 0; i < CONV1_WEIGHTS; i++) myConv1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c1std;
  for (int i = 0; i < CONV1_FILTERS; i++) myConv1_b[i] = 0;

  float d1std = sqrt(2.0f / COMBINED_SIZE);
  for (int i = 0; i < DENSE1_WEIGHTS; i++) myDense1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * d1std;
  for (int i = 0; i < DENSE1_SIZE; i++) myDense1_b[i] = 0;

  float d2std = sqrt(2.0f / DENSE1_SIZE);
  for (int i = 0; i < DENSE2_WEIGHTS; i++) myDense2_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * d2std;
  for (int i = 0; i < DENSE2_SIZE; i++) myDense2_b[i] = 0;

  float ostd = sqrt(2.0f / DENSE2_SIZE);
  for (int i = 0; i < OUTPUT_WEIGHTS; i++) myOutput_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * ostd;
  for (int i = 0; i < NUM_CLASSES; i++) myOutput_b[i] = 0;

  Serial.println("He-init random weights set");
  myCoreMemoryReady = true;
  return true;
}

bool myAllocateTrainingMemory() {
  Serial.println("\n=== Allocating Training-Only Memory (grad + Adam) ===");
  bool ok = true;
  #define MY_ALLOC_GT(ptr, size, name) \
    ptr = (float*)ps_malloc((size) * sizeof(float)); \
    ok &= myCheckAlloc(ptr, name, (size) * sizeof(float)); \
    if (ptr) memset(ptr, 0, (size) * sizeof(float));

  MY_ALLOC_GT(myConv1_w_grad, CONV1_WEIGHTS, "myConv1_w_grad");
  MY_ALLOC_GT(myConv1_b_grad, CONV1_FILTERS, "myConv1_b_grad");
  MY_ALLOC_GT(myDense1_w_grad, DENSE1_WEIGHTS, "myDense1_w_grad");
  MY_ALLOC_GT(myDense1_b_grad, DENSE1_SIZE, "myDense1_b_grad");
  MY_ALLOC_GT(myDense2_w_grad, DENSE2_WEIGHTS, "myDense2_w_grad");
  MY_ALLOC_GT(myDense2_b_grad, DENSE2_SIZE, "myDense2_b_grad");
  MY_ALLOC_GT(myOutput_w_grad, OUTPUT_WEIGHTS, "myOutput_w_grad");
  MY_ALLOC_GT(myOutput_b_grad, NUM_CLASSES, "myOutput_b_grad");

  MY_ALLOC_GT(myConv1_w_m, CONV1_WEIGHTS, "myConv1_w_m");
  MY_ALLOC_GT(myConv1_w_v, CONV1_WEIGHTS, "myConv1_w_v");
  MY_ALLOC_GT(myConv1_b_m, CONV1_FILTERS, "myConv1_b_m");
  MY_ALLOC_GT(myConv1_b_v, CONV1_FILTERS, "myConv1_b_v");
  MY_ALLOC_GT(myDense1_w_m, DENSE1_WEIGHTS, "myDense1_w_m");
  MY_ALLOC_GT(myDense1_w_v, DENSE1_WEIGHTS, "myDense1_w_v");
  MY_ALLOC_GT(myDense1_b_m, DENSE1_SIZE, "myDense1_b_m");
  MY_ALLOC_GT(myDense1_b_v, DENSE1_SIZE, "myDense1_b_v");
  MY_ALLOC_GT(myDense2_w_m, DENSE2_WEIGHTS, "myDense2_w_m");
  MY_ALLOC_GT(myDense2_w_v, DENSE2_WEIGHTS, "myDense2_w_v");
  MY_ALLOC_GT(myDense2_b_m, DENSE2_SIZE, "myDense2_b_m");
  MY_ALLOC_GT(myDense2_b_v, DENSE2_SIZE, "myDense2_b_v");
  MY_ALLOC_GT(myOutput_w_m, OUTPUT_WEIGHTS, "myOutput_w_m");
  MY_ALLOC_GT(myOutput_w_v, OUTPUT_WEIGHTS, "myOutput_w_v");
  MY_ALLOC_GT(myOutput_b_m, NUM_CLASSES, "myOutput_b_m");
  MY_ALLOC_GT(myOutput_b_v, NUM_CLASSES, "myOutput_b_v");

  MY_ALLOC_GT(myOutput_delta, NUM_CLASSES, "myOutput_delta");
  MY_ALLOC_GT(myDense2_delta, DENSE2_SIZE, "myDense2_delta");
  MY_ALLOC_GT(myDense1_delta, DENSE1_SIZE, "myDense1_delta");
  MY_ALLOC_GT(myPool1_delta, CONV1_FLAT, "myPool1_delta");
  MY_ALLOC_GT(myConv1_delta, CONV1_OUT_STEPS * CONV1_FILTERS, "myConv1_delta");
  #undef MY_ALLOC_GT

  return ok;
}

void myFreeTrainingMemory() {
  free(myConv1_w_grad); free(myConv1_b_grad); free(myDense1_w_grad); free(myDense1_b_grad);
  free(myDense2_w_grad); free(myDense2_b_grad); free(myOutput_w_grad); free(myOutput_b_grad);
  free(myConv1_w_m); free(myConv1_w_v); free(myConv1_b_m); free(myConv1_b_v);
  free(myDense1_w_m); free(myDense1_w_v); free(myDense1_b_m); free(myDense1_b_v);
  free(myDense2_w_m); free(myDense2_w_v); free(myDense2_b_m); free(myDense2_b_v);
  free(myOutput_w_m); free(myOutput_w_v); free(myOutput_b_m); free(myOutput_b_v);
  free(myOutput_delta); free(myDense2_delta); free(myDense1_delta);
  free(myPool1_delta); free(myConv1_delta);
  Serial.println("Training-only memory freed");
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 1: FORWARD / BACKWARD PASS                                         ██
// ██████████████████████████████████████████████████████████████████████████████

void myConv1DForward(float* fastInput) {
  for (int s = 0; s < CONV1_OUT_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float sum = myConv1_b[f];
      for (int k = 0; k < CONV1_KERNEL; k++) {
        for (int c = 0; c < FAST_CHANNELS; c++) {
          sum += fastInput[(s + k) * FAST_CHANNELS + c] *
                 myConv1_w[(k * FAST_CHANNELS + c) * CONV1_FILTERS + f];
        }
      }
      myConv1_output[s * CONV1_FILTERS + f] = myLeakyRelu(sum);
    }
  }
}

void myPool1Forward() {
  for (int s = 0; s < POOL1_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float a = myConv1_output[(s * 2)     * CONV1_FILTERS + f];
      float b = myConv1_output[(s * 2 + 1) * CONV1_FILTERS + f];
      myPool1_output[s * CONV1_FILTERS + f] = max(a, b);
    }
  }
}

void myDenseForward(float* input, int inSize, float* w, float* b,
                     float* output, int outSize, bool applyActivation) {
  for (int j = 0; j < outSize; j++) {
    float sum = b[j];
    for (int i = 0; i < inSize; i++) sum += input[i] * w[i * outSize + j];
    output[j] = applyActivation ? myLeakyRelu(sum) : sum;
  }
}

// normalizedPacket must already be normalized (see myNormalizePacket)
void myForwardPass(float* normalizedPacket) {
  myConv1DForward(normalizedPacket);            // uses first FAST_SIZE floats
  myPool1Forward();                              // -> myPool1_output[CONV1_FLAT]

  memcpy(myCombined, myPool1_output, CONV1_FLAT * sizeof(float));
  memcpy(myCombined + CONV1_FLAT, normalizedPacket + FAST_SIZE, STATIC_FEATURES * sizeof(float));

  myDenseForward(myCombined, COMBINED_SIZE, myDense1_w, myDense1_b, myDense1_output, DENSE1_SIZE, true);
  myDenseForward(myDense1_output, DENSE1_SIZE, myDense2_w, myDense2_b, myDense2_output, DENSE2_SIZE, true);
  myDenseForward(myDense2_output, DENSE2_SIZE, myOutput_w, myOutput_b, myFinal_output, NUM_CLASSES, false);
  mySoftmax(myFinal_output, NUM_CLASSES);
}

float myComputeLoss(int label) {
  float p = max(myFinal_output[label], 1e-7f);
  return -log(p);
}

void myAdamUpdate(float* w, float* grad, float* m, float* v, int size, float lr) {
  const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
  myAdamStep++;
  float bc1 = 1.0f - pow(beta1, myAdamStep);
  float bc2 = 1.0f - pow(beta2, myAdamStep);
  for (int i = 0; i < size; i++) {
    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
    float mHat = m[i] / bc1;
    float vHat = v[i] / bc2;
    w[i] -= lr * mHat / (sqrt(vHat) + eps);
  }
}

void myZeroGradients() {
  memset(myConv1_w_grad, 0, CONV1_WEIGHTS * sizeof(float));
  memset(myConv1_b_grad, 0, CONV1_FILTERS * sizeof(float));
  memset(myDense1_w_grad, 0, DENSE1_WEIGHTS * sizeof(float));
  memset(myDense1_b_grad, 0, DENSE1_SIZE * sizeof(float));
  memset(myDense2_w_grad, 0, DENSE2_WEIGHTS * sizeof(float));
  memset(myDense2_b_grad, 0, DENSE2_SIZE * sizeof(float));
  memset(myOutput_w_grad, 0, OUTPUT_WEIGHTS * sizeof(float));
  memset(myOutput_b_grad, 0, NUM_CLASSES * sizeof(float));
}

// normalizedPacket must be the SAME buffer already run through myForwardPass()
// (myCombined / myPool1_output / etc. reflect that forward pass's activations)
void myBackwardPass(float* normalizedPacket, int label) {
  for (int j = 0; j < NUM_CLASSES; j++)
    myOutput_delta[j] = myFinal_output[j] - (j == label ? 1.0f : 0.0f);

  for (int i = 0; i < DENSE2_SIZE; i++)
    for (int j = 0; j < NUM_CLASSES; j++)
      myOutput_w_grad[i * NUM_CLASSES + j] += myDense2_output[i] * myOutput_delta[j];
  for (int j = 0; j < NUM_CLASSES; j++) myOutput_b_grad[j] += myOutput_delta[j];

  for (int i = 0; i < DENSE2_SIZE; i++) {
    float sum = 0;
    for (int j = 0; j < NUM_CLASSES; j++) sum += myOutput_w[i * NUM_CLASSES + j] * myOutput_delta[j];
    myDense2_delta[i] = sum * myLeakyReluDeriv(myDense2_output[i]);
  }
  for (int i = 0; i < DENSE1_SIZE; i++)
    for (int j = 0; j < DENSE2_SIZE; j++)
      myDense2_w_grad[i * DENSE2_SIZE + j] += myDense1_output[i] * myDense2_delta[j];
  for (int j = 0; j < DENSE2_SIZE; j++) myDense2_b_grad[j] += myDense2_delta[j];

  for (int i = 0; i < DENSE1_SIZE; i++) {
    float sum = 0;
    for (int j = 0; j < DENSE2_SIZE; j++) sum += myDense2_w[i * DENSE2_SIZE + j] * myDense2_delta[j];
    myDense1_delta[i] = sum * myLeakyReluDeriv(myDense1_output[i]);
  }
  // Dense1 weight gradient uses ALL of COMBINED_SIZE (conv-pool part AND
  // static features both get trained connections into Dense1).
  for (int i = 0; i < COMBINED_SIZE; i++)
    for (int j = 0; j < DENSE1_SIZE; j++)
      myDense1_w_grad[i * DENSE1_SIZE + j] += myCombined[i] * myDense1_delta[j];
  for (int j = 0; j < DENSE1_SIZE; j++) myDense1_b_grad[j] += myDense1_delta[j];

  // Only the first CONV1_FLAT rows of Dense1_w need a backward delta -
  // the static-feature rows (CONV1_FLAT..COMBINED_SIZE-1) are leaf
  // inputs with nothing further upstream to update.
  for (int s = 0; s < POOL1_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      int i = s * CONV1_FILTERS + f;
      float grad = 0;
      for (int j = 0; j < DENSE1_SIZE; j++)
        grad += myDense1_w[i * DENSE1_SIZE + j] * myDense1_delta[j];
      myPool1_delta[i] = grad;

      float a = myConv1_output[(s * 2)     * CONV1_FILTERS + f];
      float b = myConv1_output[(s * 2 + 1) * CONV1_FILTERS + f];
      myConv1_delta[(s * 2)     * CONV1_FILTERS + f] = (a >= b) ? grad : 0.0f;
      myConv1_delta[(s * 2 + 1) * CONV1_FILTERS + f] = (b >  a) ? grad : 0.0f;
    }
  }

  for (int s = 0; s < CONV1_OUT_STEPS; s++) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float delta = myConv1_delta[s * CONV1_FILTERS + f] *
                    myLeakyReluDeriv(myConv1_output[s * CONV1_FILTERS + f]);
      myConv1_b_grad[f] += delta;
      for (int k = 0; k < CONV1_KERNEL; k++) {
        for (int c = 0; c < FAST_CHANNELS; c++) {
          myConv1_w_grad[(k * FAST_CHANNELS + c) * CONV1_FILTERS + f] +=
            normalizedPacket[(s + k) * FAST_CHANNELS + c] * delta;
        }
      }
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 2: BLE CENTRAL - CONNECT TO NANO 33, PULL WINDOWS                  ██
// ██████████████████████████████████████████████████████████████████████████████

NimBLEClient* myBLEClient = nullptr;
NimBLERemoteCharacteristic* myControlChar   = nullptr;
NimBLERemoteCharacteristic* myHeartbeatChar = nullptr;
NimBLERemoteCharacteristic* myBinaryChar    = nullptr;
NimBLERemoteCharacteristic* myModelCharRemote = nullptr;   // [NEW v05] Nano33's MODEL_CHAR - we WRITE to this to push a model onward
volatile bool myNano33Connected = false;

// Live heartbeat values, parsed from the Nano33's compact CSV notify
volatile bool myHeartbeatFresh = false;
float myHbAx = 0, myHbAy = 0, myHbAz = 0, myHbMic = 0, myHbTemp = 0, myHbHumidity = 0;
int   myHbProximity = 0;
unsigned long myLastHeartbeatMs = 0;

// Chunked binary window receive state
volatile bool myBinaryUploadActive = false;
volatile uint32_t myBinaryTotal = 0;
volatile uint32_t myBinaryGot   = 0;
volatile bool myWindowReady = false;   // set true once a full CAPTURE packet has landed
volatile bool myQuickReady  = false;   // set true once a QUICK packet has landed
uint8_t* myBinaryDestBuf = nullptr;    // where the current transfer is being written

// Diagnostics, kept as persistent state (not one-shot prints) so a
// Serial Monitor that connects late can still see current status via
// the periodic status line in loop() below.
bool myHbSubscribeOk = false;
bool myBinSubscribeOk = false;
uint32_t myBinaryPacketsReceived = 0;

class MyClientCallbacks : public NimBLEClientCallbacks {
  void onDisconnect(NimBLEClient* pClient, int reason) override {
    myNano33Connected = false;
    Serial.println("Nano33 disconnected");
  }
};

void myHeartbeatNotifyCB(NimBLERemoteCharacteristic* c, uint8_t* data, size_t len, bool isNotify) {
  char buf[80];
  int n = min((size_t)79, len);
  memcpy(buf, data, n);
  buf[n] = '\0';
  // format: ax,ay,az,mic,temp,hum,prox
  float v[7] = {0};
  int idx = 0;
  char* tok = strtok(buf, ",");
  while (tok && idx < 7) { v[idx++] = atof(tok); tok = strtok(nullptr, ","); }
  if (idx >= 7) {
    if (!myHeartbeatFresh && myDebugVerbose) Serial.println("Heartbeat notify: first packet received - link is alive");
    myHbAx = v[0]; myHbAy = v[1]; myHbAz = v[2]; myHbMic = v[3];
    myHbTemp = v[4]; myHbHumidity = v[5]; myHbProximity = (int)v[6];
    myHeartbeatFresh = true;
    myLastHeartbeatMs = millis();
  }
}

void myBinaryNotifyCB(NimBLERemoteCharacteristic* c, uint8_t* data, size_t len, bool isNotify) {
  myBinaryPacketsReceived++;
  if (!myBinaryUploadActive) {
    if (len == 4) {
      myBinaryTotal = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
                      ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
      myBinaryGot = 0;
      if (myBinaryTotal == PACKET_BYTES) {
        myBinaryDestBuf = myPacketBytes;
        myBinaryUploadActive = true;
      } else if (myBinaryTotal == QUICK_PACKET_BYTES) {
        myBinaryDestBuf = myQuickBytes;
        myBinaryUploadActive = true;
      } else {
        myBinaryUploadActive = false;
        Serial.printf("WARN: unexpected packet size %u (wanted %u or %u) - ignoring\n",
                      (unsigned)myBinaryTotal, (unsigned)PACKET_BYTES, (unsigned)QUICK_PACKET_BYTES);
      }
      if (myDebugVerbose) {
        Serial.printf("Binary notify: size header received = %u bytes\n", (unsigned)myBinaryTotal);
      }
    } else {
      Serial.printf("Binary notify: unexpected first packet len=%u (wanted 4-byte header) - ignoring\n", (unsigned)len);
    }
    return;
  }
  size_t room = myBinaryTotal - myBinaryGot;
  size_t n = min(len, room);
  memcpy(myBinaryDestBuf + myBinaryGot, data, n);
  myBinaryGot += n;
  if (myBinaryGot >= myBinaryTotal) {
    myBinaryUploadActive = false;
    if (myBinaryTotal == PACKET_BYTES) myWindowReady = true;
    else myQuickReady = true;
    if (myDebugVerbose) Serial.println("Binary notify: transfer complete");
  }
}

bool myBLEConnectToNano33() {
  Serial.println("Scanning for Nano33 peripheral...");
  NimBLEScan* scan = NimBLEDevice::getScan();
  scan->setActiveScan(true);
  scan->setInterval(100);
  scan->setWindow(99);
  NimBLEScanResults results = scan->getResults(5000, false);

  NimBLEAddress targetAddr;
  bool found = false;
  for (int i = 0; i < results.getCount(); i++) {
    const NimBLEAdvertisedDevice* dev = results.getDevice(i);
    if (dev->haveName() && String(dev->getName().c_str()).startsWith(MY_NANO33_NAME_PREFIX)) {
      targetAddr = dev->getAddress();
      found = true;
      Serial.printf("Found: %s  addr=%s\n", dev->getName().c_str(), targetAddr.toString().c_str());
      break;
    }
  }
  scan->clearResults();
  if (!found) {
    Serial.println("Nano33 peripheral not found - is it powered and advertising?");
    return false;
  }

  if (!myBLEClient) {
    myBLEClient = NimBLEDevice::createClient();
    myBLEClient->setClientCallbacks(new MyClientCallbacks());
  }
  if (!myBLEClient->connect(targetAddr)) {
    Serial.println("Connect failed");
    return false;
  }

  NimBLERemoteService* svc = myBLEClient->getService(MY_BLE_SERVICE_UUID);
  if (!svc) { Serial.println("Service not found"); myBLEClient->disconnect(); return false; }

  myControlChar   = svc->getCharacteristic(MY_BLE_CONTROL_CHAR_UUID);
  myHeartbeatChar = svc->getCharacteristic(MY_BLE_HEARTBEAT_CHAR_UUID);
  myBinaryChar    = svc->getCharacteristic(MY_BLE_BINARY_CHAR_UUID);
  if (!myControlChar || !myHeartbeatChar || !myBinaryChar) {
    Serial.println("Characteristics not found");
    myBLEClient->disconnect();
    return false;
  }

  // [NEW v05] optional - only present on a nano33-v05+ peripheral. Not
  // fatal if missing so this sketch still connects fine to older
  // nano33-v0x firmware; myPushModelToNano33() just reports "not
  // available" instead of crashing.
  myModelCharRemote = svc->getCharacteristic(MY_BLE_MODEL_CHAR_UUID);
  Serial.printf("Nano33 model-receive characteristic: %s\n", myModelCharRemote ? "found" : "not present (old Nano33 firmware?)");

  bool hbOk = false, binOk = false;
  if (myHeartbeatChar->canNotify()) {
    hbOk = myHeartbeatChar->subscribe(true, myHeartbeatNotifyCB);
  } else {
    Serial.println("WARN: heartbeat characteristic does not support notify");
  }
  if (myBinaryChar->canIndicate()) {
    binOk = myBinaryChar->subscribe(false, myBinaryNotifyCB);   // false = indications, not notifications
  } else if (myBinaryChar->canNotify()) {
    Serial.println("WARN: binary char only supports Notify, not Indicate - transfer may drop packets");
    binOk = myBinaryChar->subscribe(true, myBinaryNotifyCB);
  } else {
    Serial.println("WARN: binary characteristic does not support notify or indicate");
  }
  myHbSubscribeOk = hbOk;
  myBinSubscribeOk = binOk;
  Serial.printf("Heartbeat subscribe: %s   Binary subscribe: %s\n",
                hbOk ? "OK" : "FAILED", binOk ? "OK" : "FAILED");
  if (!binOk) {
    Serial.println("Binary notify subscription failed - captures will always time out.");
    Serial.println("Try 'r' to reconnect, or power-cycle both boards.");
  }

  myNano33Connected = true;
  Serial.println("Connected to Nano33 and subscribed.");
  return true;
}

// Blocking request-response: write CAPTURE, wait for the chunked window
// to fully arrive (or time out). Result lands in myPacketBytes/myPacketFloats.
bool myRequestWindowBlocking(unsigned long timeoutMs) {
  if (!myNano33Connected || !myControlChar) {
    Serial.println("Request aborted: not connected to Nano33 (or control characteristic missing)");
    return false;
  }
  myWindowReady = false;
  myBinaryUploadActive = false;
  uint32_t packetsBefore = myBinaryPacketsReceived;
  const char* cmd = "CAPTURE";
  myControlChar->writeValue((uint8_t*)cmd, strlen(cmd), true);

  unsigned long start = millis();
  unsigned long lastReport = start;
  while (!myWindowReady && millis() - start < timeoutMs) {
    delay(10);
    if (myDebugVerbose && millis() - lastReport > 500) {
      lastReport = millis();
      Serial.printf("  ...waiting (%lums) - binary notify packets seen so far: %u, got=%u/%u bytes\n",
                    millis() - start, (unsigned)(myBinaryPacketsReceived - packetsBefore),
                    (unsigned)myBinaryGot, (unsigned)myBinaryTotal);
    }
  }
  if (!myWindowReady && myBinaryPacketsReceived == packetsBefore) {
    Serial.println("  DIAGNOSIS: zero binary notify packets arrived at all.");
    Serial.println("  This points to the notify subscription, not the Nano33 (which is sampling/sending fine).");
    Serial.printf("  Heartbeat subscribe was: %s   Binary subscribe was: %s\n",
                  myHbSubscribeOk ? "OK" : "FAILED", myBinSubscribeOk ? "OK" : "FAILED");
  }
  return myWindowReady;
}

// Fast path: single instant reading, no 1-second sampling wait.
// Much lower latency than myRequestWindowBlocking() - use for
// continuous/live inference where speed matters more than capturing
// real motion over a full second.
bool myRequestQuickBlocking(unsigned long timeoutMs) {
  if (!myNano33Connected || !myControlChar) return false;
  myQuickReady = false;
  myBinaryUploadActive = false;
  const char* cmd = "QUICK";
  myControlChar->writeValue((uint8_t*)cmd, strlen(cmd), true);

  unsigned long start = millis();
  while (!myQuickReady && millis() - start < timeoutMs) {
    delay(5);
  }
  return myQuickReady;
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 2.5: BLE PERIPHERAL SERVER - lets the webpage connect      [NEW v05] ██
// ██  directly to THIS board and drive the session, same 4 v04 UUIDs +2 new    ██
// ██████████████████████████████████████████████████████████████████████████████
//
// This runs ALONGSIDE the NimBLEClient role above (this board stays
// central to the Nano33 the whole time). Browser commands land on
// myServerControlChar and are only ever turned into a "please do X"
// flag here - the actual work happens in loop(), never inside the
// BLE callback itself, so a long blocking call like myActionTrain()
// doesn't stall the NimBLE host task.

NimBLEServer* myServer = nullptr;
NimBLECharacteristic* myServerControlChar   = nullptr;
NimBLECharacteristic* myServerHeartbeatChar = nullptr;
NimBLECharacteristic* myServerBinaryChar    = nullptr;
NimBLECharacteristic* myServerModelChar     = nullptr;   // browser -> ESP32 model push
NimBLECharacteristic* myServerResultChar    = nullptr;   // ESP32 -> browser inference result
volatile bool myBrowserConnected = false;

// Flags set by the control-char BLE callback, consumed in loop()
volatile bool myBrowserWantsCapture     = false;
volatile bool myBrowserWantsQuick       = false;
volatile bool myBrowserWantsTrain       = false;
volatile bool myBrowserWantsInfer       = false;
volatile bool myBrowserWantsQuickInfer  = false;
volatile bool myBrowserWantsPushToNano  = false;
volatile bool myBrowserWantsPushModel   = false;   // send THIS board's model out to the browser (over binary char)
volatile bool myBrowserWantsReconnect   = false;
volatile bool myBrowserWantsCalibrate   = false;
volatile bool myBrowserWantsStatus      = false;

// Model receive state (browser -> ESP32), mirrors nano33-v05.ino's
// myOnModelWrite() state machine exactly.
volatile bool myServerModelRxActive = false;
volatile uint32_t myServerModelRxTotal = 0;
volatile uint32_t myServerModelRxGot   = 0;

class MyServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* s, NimBLEConnInfo& info) override {
    myBrowserConnected = true;
    Serial.println("Browser connected to ESP32 peripheral role");
  }
  void onDisconnect(NimBLEServer* s, NimBLEConnInfo& info, int reason) override {
    myBrowserConnected = false;
    Serial.println("Browser disconnected - resuming advertising");
    NimBLEDevice::startAdvertising();
  }
};

class MyServerControlCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& info) override {
    std::string v = c->getValue();
    String cmd = String(v.c_str());
    cmd.trim();
    Serial.print("Browser control RX: \"");
    Serial.print(cmd);
    Serial.println("\"");

    if (cmd == "CAPTURE")          myBrowserWantsCapture = true;
    else if (cmd == "QUICK")       myBrowserWantsQuick = true;
    else if (cmd == "TRAIN")       myBrowserWantsTrain = true;
    else if (cmd == "INFER")       myBrowserWantsInfer = true;
    else if (cmd == "QUICKINFER")  myBrowserWantsQuickInfer = true;
    else if (cmd == "PUSHTONANO")  myBrowserWantsPushToNano = true;
    else if (cmd == "PUSHMODEL")   myBrowserWantsPushModel = true;
    else if (cmd == "r")           myBrowserWantsReconnect = true;
    else if (cmd == "k")           myBrowserWantsCalibrate = true;
    else if (cmd == "s")           myBrowserWantsStatus = true;
  }
};

// Receives a browser-trained model package, chunked exactly like the
// Nano33's myOnModelWrite(): first write = 4-byte LE length header,
// then raw chunks until the total arrives.
class MyServerModelCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& info) override {
    std::string v = c->getValue();
    const uint8_t* data = (const uint8_t*)v.data();
    size_t len = v.size();

    if (!myServerModelRxActive) {
      if (len == 4) {
        myServerModelRxTotal = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
                                ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
        if (myServerModelRxTotal != MY_MODEL_PACKAGE_BYTES) {
          Serial.printf("ERROR: browser model package wrong size %u (wanted %u)\n",
                        (unsigned)myServerModelRxTotal, (unsigned)MY_MODEL_PACKAGE_BYTES);
          return;
        }
        myServerModelRxGot = 0;
        myServerModelRxActive = true;
        Serial.println("Browser model push starting...");
      }
      return;
    }
    uint32_t room = myServerModelRxTotal - myServerModelRxGot;
    uint32_t n = min((uint32_t)len, room);
    memcpy(myModelPackageBuf + myServerModelRxGot, data, n);
    myServerModelRxGot += n;
    if (myServerModelRxGot >= myServerModelRxTotal) {
      myServerModelRxActive = false;
      Serial.println("Browser model push complete - importing...");
      myImportModelPackage();
    }
  }
};

void myStartBLEPeripheralServer() {
  myServer = NimBLEDevice::createServer();
  myServer->setCallbacks(new MyServerCallbacks());

  NimBLEService* svc = myServer->createService(MY_BLE_SERVICE_UUID);
  myServerControlChar   = svc->createCharacteristic(MY_BLE_CONTROL_CHAR_UUID, NIMBLE_PROPERTY::WRITE);
  myServerHeartbeatChar = svc->createCharacteristic(MY_BLE_HEARTBEAT_CHAR_UUID, NIMBLE_PROPERTY::NOTIFY);
  myServerBinaryChar    = svc->createCharacteristic(MY_BLE_BINARY_CHAR_UUID, NIMBLE_PROPERTY::NOTIFY);   // [CHANGED v14.2] was INDICATE - see myServerSendBinary()
  myServerModelChar     = svc->createCharacteristic(MY_BLE_MODEL_CHAR_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  myServerResultChar    = svc->createCharacteristic(MY_BLE_RESULT_CHAR_UUID, NIMBLE_PROPERTY::NOTIFY);

  myServerControlChar->setCallbacks(new MyServerControlCallbacks());
  myServerModelChar->setCallbacks(new MyServerModelCallbacks());

  svc->start();

  // ======================================================
  // [FIXED v08] Legacy BLE advertising packets max out at 31 bytes.
  // A 128-bit service UUID (18 bytes) + this board's name (17 bytes)
  // + flags (3 bytes) = 38 bytes, which doesn't fit. adv->start() was
  // failing SILENTLY (its return value was never checked), so the
  // board never actually advertised - hence "ESP32 doesn't show up in
  // the browser's connection window" even though nothing crashed.
  // The webpage's requestDevice() filters by namePrefix, not by
  // advertised service UUID (optionalServices is only a post-connect
  // permission list, not a scan filter) - so the service UUID doesn't
  // need to be broadcast at all. Advertising just the name fits easily
  // (~20 bytes) and the browser still finds MY_BLE_SERVICE_UUID fine
  // via normal GATT discovery once connected.
  // ======================================================
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->setName(MY_ESP32_PERIPHERAL_NAME);
  bool advOk = adv->start();

  if (!advOk) {
    // Surfaced instead of silently failing, in case something else
    // (radio busy, duplicate start, low-level NimBLE error) trips this
    // again in the future.
    Serial.println("ERROR: BLE advertising failed to start! The ESP32 will NOT be visible to the browser.");
  } else {
    Serial.print("BLE peripheral role advertising as \"");
    Serial.print(MY_ESP32_PERIPHERAL_NAME);
    Serial.println("\" - webpage (index-v05.html) can now connect directly to this board.");
  }
}

// Relay a raw sensor buffer out to the browser over myServerBinaryChar,
// using the same 4-byte-header + chunk pattern used everywhere else in
// this protocol family.
// [CHANGED v14.2] "Fetch ESP32 Model to Browser" was STILL stalling at
// the EXACT same byte count (2968) even after v14.1's much more patient
// retry loop - identical down to the byte, across two runs with
// different retry timing. That rules out ordinary congestion (which
// would vary run to run); it points at a hard limit specific to
// INDICATE, not a queue that just needed more time. This matches a
// known class of Web Bluetooth / OS BLE stack limitation around
// confirmed indications on sustained transfers (particularly reported
// on Windows) - the ATT-level confirmation INDICATE relies on can
// silently stop being serviced by the OS past a certain data volume,
// which no amount of retrying on the ESP32 side can fix, because the
// ESP32 genuinely never gets a confirmation back to retry against.
// Switched myServerBinaryChar (see its createCharacteristic() call
// above) from INDICATE to NOTIFY - unconfirmed, no OS-level
// acknowledgement to get stuck waiting on. The trade-off is losing
// indicate's guaranteed-delivery property, so an occasional dropped
// notify is possible in principle; in exchange the transfer isn't
// gated on a confirmation channel that's demonstrated it doesn't work
// reliably for this data volume. myNotifyWithRetry() below still
// retries on a false return (notify() can still fail locally if the
// ESP32's own TX queue is briefly full - that part of v14.1's fix was
// sound and is kept), and the browser's own receivedBytes-vs-
// expectedBytes check plus the stale-transfer watchdog still catch
// and clearly report an incomplete transfer if anything does slip
// through, rather than silently doing something wrong with it.
// If this still isn't reliable enough in practice, index-v14.html also
// now has a "Paste Model From Serial" fallback (myExportModelAsHex(),
// 'e' Serial command below) that sends the model over USB Serial
// instead of BLE entirely - immune to any BLE-side limitation, at the
// cost of a manual copy/paste.
#define MY_NOTIFY_MAX_WAIT_MS   3000   // [NEW v14.1] total time willing to wait for ONE chunk to be accepted
#define MY_NOTIFY_RETRY_MS      40     // [NEW v14.1] pause between retry attempts

// [CHANGED v14.2] was myIndicateWithRetry() / ch->indicate() - see the
// header comment above for why this now calls notify() instead. Still
// retries on a false return (a locally-full TX queue), the browser
// disconnecting, or MY_NOTIFY_MAX_WAIT_MS being used up. Returns false
// only when it's given up for real.
bool myNotifyWithRetry(NimBLECharacteristic* ch) {
  bool sent = ch->notify();
  unsigned long start = millis();
  while (!sent && myBrowserConnected && (millis() - start) < MY_NOTIFY_MAX_WAIT_MS) {
    delay(MY_NOTIFY_RETRY_MS);
    sent = ch->notify();
  }
  return sent;
}

void myServerSendBinary(const uint8_t* raw, uint32_t totalBytes) {
  if (!myServerBinaryChar || !myBrowserConnected) return;
  uint8_t header[4] = {
    (uint8_t)(totalBytes & 0xFF), (uint8_t)((totalBytes >> 8) & 0xFF),
    (uint8_t)((totalBytes >> 16) & 0xFF), (uint8_t)((totalBytes >> 24) & 0xFF)
  };
  myServerBinaryChar->setValue(header, 4);
  bool ok = myNotifyWithRetry(myServerBinaryChar);   // [CHANGED v14.2]
  delay(15);

  const int chunkSize = 180;
  int chunkNum = 0;
  int totalChunks = (totalBytes + chunkSize - 1) / chunkSize;
  unsigned long transferStart = millis();
  for (uint32_t off = 0; ok && off < totalBytes; off += chunkSize) {
    chunkNum++;
    int n = min((uint32_t)chunkSize, totalBytes - off);
    myServerBinaryChar->setValue(raw + off, n);
    bool sent = myNotifyWithRetry(myServerBinaryChar);   // [CHANGED v14.2]
    if (!sent) {
      Serial.printf("ERROR: binary send to browser failed at chunk %d/%d (%u/%u bytes sent, gave up after %lums) - aborting transfer. Browser %s.\n",
                    chunkNum, totalChunks, (unsigned)off, (unsigned)totalBytes, MY_NOTIFY_MAX_WAIT_MS,
                    myBrowserConnected ? "still connected - link is just stuck" : "disconnected mid-transfer");
      ok = false;
      break;
    }
    if (myDebugVerbose && (chunkNum % 20 == 0)) {   // [NEW v14.1] quiet progress ping for the long model transfer
      Serial.printf("  ...binary send to browser: chunk %d/%d (%u/%u bytes)\n", chunkNum, totalChunks, (unsigned)off + n, (unsigned)totalBytes);
    }
    delay(15);   // small pacing gap even on success, to keep the queue from refilling instantly
  }
  if (ok) {
    Serial.printf("Binary send to browser OK - %u bytes in %lums\n", (unsigned)totalBytes, millis() - transferStart);
  } else {
    Serial.println("Binary send to browser FAILED - the browser will be left waiting; try the fetch/push again, or use the 'e' Serial command + \"Paste Model From Serial\" on the webpage as a BLE-free fallback.");
  }
}

// ======================================================
// [NEW v14.2] PASTE-FROM-SERIAL MODEL EXPORT - a BLE-free fallback for
// getting this board's trained model into the browser, for cases where
// "Fetch ESP32 Model to Browser" still isn't reliable on a given
// machine/OS/browser combination. Dumps the model package as plain hex
// text over USB Serial, wrapped in BEGIN/END markers; the matching
// "Paste Model From Serial" box on index-v14.html strips the markers,
// decodes the hex, and loads it exactly like a completed BLE fetch
// would. USB Serial has none of BLE's packet-size/queue/confirmation
// limitations, so this always works as long as the board is plugged
// in and its model is trained.
// ======================================================
void myExportModelAsHex() {
  if (!myWeightsTrained) {
    Serial.println("No trained weights on this board yet - train first ('t'), or push a model in from the Nano33/webpage.");
    return;
  }
  myExportModelPackage();   // fills myModelPackageBuf, MY_MODEL_PACKAGE_BYTES long
  Serial.println("\n===MODEL_EXPORT_BEGIN===");
  const char hexDigits[] = "0123456789abcdef";
  char lineBuf[121];
  int col = 0;
  for (uint32_t i = 0; i < MY_MODEL_PACKAGE_BYTES; i++) {
    lineBuf[col++] = hexDigits[(myModelPackageBuf[i] >> 4) & 0xF];
    lineBuf[col++] = hexDigits[myModelPackageBuf[i] & 0xF];
    if (col >= 120) { lineBuf[col] = '\0'; Serial.println(lineBuf); col = 0; }
  }
  if (col > 0) { lineBuf[col] = '\0'; Serial.println(lineBuf); }
  Serial.println("===MODEL_EXPORT_END===");
  Serial.println("Select everything between (not including) the BEGIN/END marker lines above, copy it,");
  Serial.println("and paste it into the \"Paste Model From Serial\" box on the webpage, under this board's panel.");
}

// Broadcast an inference result to the browser (index-based CSV, same
// format as nano33-v05.ino's myRunInferenceAndReport, so a single
// browser-side parser handles results from either board).
void myServerSendResult(int pred, float* probs, int numClasses) {
  if (!myServerResultChar || !myBrowserConnected) return;
  char out[96];
  int n = snprintf(out, sizeof(out), "%d,%.3f", pred, probs[pred]);
  for (int j = 0; j < numClasses && n < (int)sizeof(out) - 8; j++) {
    n += snprintf(out + n, sizeof(out) - n, ",%.3f", probs[j]);
  }
  myServerResultChar->setValue((uint8_t*)out, n);
  myServerResultChar->notify();
}

// [NEW v14] Shared Serial result-line formatter - identical text on
// this board's Infer AND Quick Infer (previously two different
// formats), and identical to nano33-v14.ino's own myPrintResultLine().
// See that file's header comment for the exact format rationale.
void myPrintResultLine(int pred, float* probs, String* labels, int numClasses) {
  Serial.print("On-device result: ");
  String predLabel = (pred >= 0 && pred < numClasses && labels[pred].length() > 0)
                        ? labels[pred] : (String("class#") + pred);
  Serial.print(predLabel);
  Serial.print(" (");
  for (int j = 0; j < numClasses; j++) {
    String lbl = labels[j].length() > 0 ? labels[j] : (String("class#") + j);
    Serial.print(lbl);
    Serial.print("=");
    Serial.print(probs[j] * 100.0f, 0);
    Serial.print("%,");
    if (j < numClasses - 1) Serial.print(" ");
  }
  Serial.println(")");
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 3: CALIBRATION                                                     ██
// ██████████████████████████████████████████████████████████████████████████████

void myCalibrate() {
  if (mySDavailable && SD.exists("/header/myFusionCalib.bin")) {
    File f = SD.open("/header/myFusionCalib.bin", FILE_READ);
    if (f && f.size() == FAST_CHANNELS * 2 * 4) {
      f.read((uint8_t*)myFastMean, FAST_CHANNELS * 4);
      f.read((uint8_t*)myFastStd,  FAST_CHANNELS * 4);
      f.close();
      Serial.println("Calibration loaded from SD");
      return;
    }
    if (f) f.close();
  }

  if (!myNano33Connected) {
    Serial.println("Skipping calibration - Nano33 not connected. Using defaults.");
    return;
  }

  Serial.println("Calibrating - keep sensor/machine stationary for one window...");
  u8g2.firstPage();
  do { u8g2.setFont(u8g2_font_5x7_tf); u8g2.drawStr(0, 10, "Calibrating..."); u8g2.drawStr(0, 22, "Keep still!"); } while (u8g2.nextPage());

  if (!myRequestWindowBlocking(MY_CAPTURE_TIMEOUT_MS)) {
    Serial.println("Calibration capture failed - using default mean/std");
    return;
  }

  for (int c = 0; c < FAST_CHANNELS; c++) {
    float sum = 0, sum2 = 0;
    for (int t = 0; t < WINDOW_TIMESTEPS; t++) {
      float v = myPacketFloats[t * FAST_CHANNELS + c];
      sum += v; sum2 += v * v;
    }
    float mean = sum / WINDOW_TIMESTEPS;
    float var = (sum2 / WINDOW_TIMESTEPS) - (mean * mean);
    myFastMean[c] = mean;
    myFastStd[c]  = max(sqrt(max(var, 0.0f)), 0.01f);
  }
  Serial.println("Calibration done (single-window estimate).");

  if (mySDavailable) {
    if (!SD.exists("/header")) SD.mkdir("/header");
    File f = SD.open("/header/myFusionCalib.bin", FILE_WRITE);
    if (f) {
      f.write((uint8_t*)myFastMean, FAST_CHANNELS * 4);
      f.write((uint8_t*)myFastStd,  FAST_CHANNELS * 4);
      f.close();
      Serial.println("Calibration saved to SD");
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 4: WEIGHT SAVE / LOAD / EXPORT                                     ██
// ██████████████████████████████████████████████████████████████████████████████

void myExportHeader() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myFusionWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_FUSION_MODEL_H\n#define MY_FUSION_MODEL_H");
  file.println("// Uncomment in main sketch:  #define USE_BAKED_WEIGHTS");
  auto myDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = { ", name);
    for (int i = 0; i < size; i++) {
      file.print(data[i], 6); file.print("f");
      if (i < size - 1) file.print(", ");
      if ((i + 1) % 8 == 0) file.println();
    }
    file.println(" };");
  };
  myDump("myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDump("myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDump("myModel_dense1_w", myDense1_w, DENSE1_WEIGHTS);
  myDump("myModel_dense1_b", myDense1_b, DENSE1_SIZE);
  myDump("myModel_dense2_w", myDense2_w, DENSE2_WEIGHTS);
  myDump("myModel_dense2_b", myDense2_b, DENSE2_SIZE);
  myDump("myModel_output_w", myOutput_w, OUTPUT_WEIGHTS);
  myDump("myModel_output_b", myOutput_b, NUM_CLASSES);
  file.println("#endif");
  file.close();
  Serial.println("Header exported to /header/myFusionWeights.h");
}

bool myLoadWeights() {
  if (!mySDavailable) return false;
  if (!SD.exists("/header/myFusionWeights.bin")) return false;
  File f = SD.open("/header/myFusionWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
  f.read((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
  f.read((uint8_t*)myDense1_w, DENSE1_WEIGHTS * 4);
  f.read((uint8_t*)myDense1_b, DENSE1_SIZE    * 4);
  f.read((uint8_t*)myDense2_w, DENSE2_WEIGHTS * 4);
  f.read((uint8_t*)myDense2_b, DENSE2_SIZE    * 4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
  f.read((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
  f.close();
  Serial.println("Weights loaded from SD");
  myWeightsTrained = true;
  return true;
}

void mySaveWeights() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myFusionWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
    f.write((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
    f.write((uint8_t*)myDense1_w, DENSE1_WEIGHTS * 4);
    f.write((uint8_t*)myDense1_b, DENSE1_SIZE    * 4);
    f.write((uint8_t*)myDense2_w, DENSE2_WEIGHTS * 4);
    f.write((uint8_t*)myDense2_b, DENSE2_SIZE    * 4);
    f.write((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
    f.write((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
    f.close();
    Serial.println("Weights saved to SD");
  }
  myExportHeader();
}

// ======================================================
// MODEL PACKAGE EXPORT/IMPORT [NEW v05] - bundles this board's live
// weight arrays + calibration + class count into the wire format both
// the Nano33 and the webpage understand (see MODEL PACKAGE LAYOUT
// comment near the UUID defines above). Used for:
//   - browser "PUSHMODEL": send this board's current weights out to
//     the browser for download/inspection
//   - browser "PUSHTONANO" / myPushModelToNano33(): forward this
//     board's current weights onward to the Nano33 over the existing
//     central link
// A static buffer is used (not PSRAM) since it's only needed
// transiently during a BLE transfer, not during training.
// ======================================================
//uint8_t myModelPackageBuf[MY_MODEL_PACKAGE_BYTES];  must be forward declared

void myExportModelPackage() {
  float* f = (float*)myModelPackageBuf;
  int idx = 0;
  f[idx++] = (float)NUM_CLASSES;
  memcpy(f + idx, myConv1_w,  CONV1_WEIGHTS  * 4); idx += CONV1_WEIGHTS;
  memcpy(f + idx, myConv1_b,  CONV1_FILTERS  * 4); idx += CONV1_FILTERS;
  memcpy(f + idx, myDense1_w, DENSE1_WEIGHTS * 4); idx += DENSE1_WEIGHTS;
  memcpy(f + idx, myDense1_b, DENSE1_SIZE    * 4); idx += DENSE1_SIZE;
  memcpy(f + idx, myDense2_w, DENSE2_WEIGHTS * 4); idx += DENSE2_WEIGHTS;
  memcpy(f + idx, myDense2_b, DENSE2_SIZE    * 4); idx += DENSE2_SIZE;
  memcpy(f + idx, myOutput_w, OUTPUT_WEIGHTS * 4); idx += OUTPUT_WEIGHTS;
  memcpy(f + idx, myOutput_b, NUM_CLASSES    * 4); idx += NUM_CLASSES;
  memcpy(f + idx, myFastMean, FAST_CHANNELS  * 4); idx += FAST_CHANNELS;
  memcpy(f + idx, myFastStd,  FAST_CHANNELS  * 4); idx += FAST_CHANNELS;
}

// Loads a received model package (already sitting in myModelPackageBuf,
// MY_MODEL_PACKAGE_BYTES long) into this board's LIVE weight arrays -
// used when the browser pushes a browser-trained model to the ESP32.
// Rejects a mismatched class count rather than corrupting memory,
// since this board's arrays are fixed-size at compile-time NUM_CLASSES
// (unlike the Nano33, which supports a range up to MY_MAX_CLASSES).
bool myImportModelPackage() {
  float* f = (float*)myModelPackageBuf;
  int idx = 0;
  int pkgClasses = (int)(f[idx++] + 0.5f);
  if (pkgClasses != NUM_CLASSES) {
    Serial.printf("ERROR: received model has %d classes, this board expects %d - rejected\n", pkgClasses, NUM_CLASSES);
    return false;
  }
  memcpy(myConv1_w,  f + idx, CONV1_WEIGHTS  * 4); idx += CONV1_WEIGHTS;
  memcpy(myConv1_b,  f + idx, CONV1_FILTERS  * 4); idx += CONV1_FILTERS;
  memcpy(myDense1_w, f + idx, DENSE1_WEIGHTS * 4); idx += DENSE1_WEIGHTS;
  memcpy(myDense1_b, f + idx, DENSE1_SIZE    * 4); idx += DENSE1_SIZE;
  memcpy(myDense2_w, f + idx, DENSE2_WEIGHTS * 4); idx += DENSE2_WEIGHTS;
  memcpy(myDense2_b, f + idx, DENSE2_SIZE    * 4); idx += DENSE2_SIZE;
  memcpy(myOutput_w, f + idx, OUTPUT_WEIGHTS * 4); idx += OUTPUT_WEIGHTS;
  memcpy(myOutput_b, f + idx, NUM_CLASSES    * 4); idx += NUM_CLASSES;
  memcpy(myFastMean, f + idx, FAST_CHANNELS  * 4); idx += FAST_CHANNELS;
  memcpy(myFastStd,  f + idx, FAST_CHANNELS  * 4); idx += FAST_CHANNELS;
  myWeightsTrained = true;
  Serial.println("Browser-trained model imported and now live on this board.");
  mySaveWeights();   // persist so it survives a reboot, same as an on-device Train run would
  return true;
}

// Push whatever model is currently live on THIS board (self-trained or
// just imported from the browser) onward to the Nano33, using the same
// 4-byte-length-header + chunk pattern the Nano33 uses to send binary
// windows back, just reversed in direction (we write, it receives).
bool myPushModelToNano33() {
  if (!myNano33Connected || !myModelCharRemote) {
    Serial.println("Cannot push model - Nano33 not connected or its model characteristic is unavailable (old firmware?)");
    return false;
  }
  if (!myWeightsTrained) {
    Serial.println("Cannot push model - no trained weights on this board yet");
    return false;
  }
  myExportModelPackage();

  uint32_t total = MY_MODEL_PACKAGE_BYTES;
  uint8_t header[4] = {
    (uint8_t)(total & 0xFF), (uint8_t)((total >> 8) & 0xFF),
    (uint8_t)((total >> 16) & 0xFF), (uint8_t)((total >> 24) & 0xFF)
  };
  bool ok = myModelCharRemote->writeValue(header, 4, true);   // true = with response, wait for ack
  const int chunkSize = 180;
  for (uint32_t off = 0; ok && off < total; off += chunkSize) {
    int n = min((uint32_t)chunkSize, total - off);
    ok = myModelCharRemote->writeValue(myModelPackageBuf + off, n, true);
  }
  Serial.printf("Push model to Nano33: %s (%u bytes)\n", ok ? "OK" : "FAILED", (unsigned)total);
  return ok;
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 5: DATA COLLECTION (pull a window from Nano33, save raw to SD)     ██
// ██████████████████████████████████████████████████████████████████████████████

int myCountSamples(int classIdx) {
  if (!mySDavailable) return 0;
  String path = "/fusion/" + myClassLabels[classIdx];
  File root = SD.open(path);
  if (!root) return 0;
  int count = 0;
  while (File f = root.openNextFile()) {
    if (!f.isDirectory() && String(f.name()).endsWith(".dat")) count++;
    f.close();
  }
  root.close();
  return count;
}

bool myCaptureAndSave(int classIdx) {
  if (!mySDavailable) { Serial.println("No SD card - cannot save"); return false; }
  if (!myNano33Connected) { Serial.println("Nano33 not connected"); return false; }

  Serial.println("Requesting window from Nano33...");
  if (!myRequestWindowBlocking(MY_CAPTURE_TIMEOUT_MS)) {
    Serial.println("Capture timed out - no window received");
    return false;
  }

  String folderPath = "/fusion/" + myClassLabels[classIdx];
  if (!SD.exists("/fusion")) SD.mkdir("/fusion");
  if (!SD.exists(folderPath)) SD.mkdir(folderPath);

  int sampleNum = myCountSamples(classIdx);
  String filePath = folderPath + "/s" + String(sampleNum) + ".dat";
  File f = SD.open(filePath, FILE_WRITE);
  if (!f) { Serial.println("ERROR: cannot open file for writing"); return false; }
  f.write(myPacketBytes, PACKET_BYTES);
  f.close();
  Serial.printf("Saved: %s (%d bytes)\n", filePath.c_str(), PACKET_BYTES);
  return true;
}

bool myLoadSampleFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if (!f) return false;
  if (f.size() != PACKET_BYTES) { Serial.printf("WARN: %s wrong size (%u)\n", path, (unsigned)f.size()); f.close(); return false; }
  f.read((uint8_t*)buf, PACKET_BYTES);
  f.close();
  myNormalizePacket(buf);
  return true;
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 6: TRAIN / INFER                                                   ██
// ██████████████████████████████████████████████████████████████████████████████

void myActionTrain() {
  if (!mySDavailable) { Serial.println("No SD - cannot train"); return; }

  myTrainingData.clear();
  int classCounts[NUM_CLASSES] = {};
  for (int c = 0; c < NUM_CLASSES; c++) {
    String path = "/fusion/" + myClassLabels[c];
    File root = SD.open(path);
    if (!root) continue;
    while (File file = root.openNextFile()) {
      String name = file.name();
      if (!file.isDirectory() && name.endsWith(".dat")) {
        myTrainingData.push_back({path + "/" + name, c});
        classCounts[c]++;
      }
      file.close();
    }
    root.close();
  }

  Serial.println("\n=== Training ===");
  for (int c = 0; c < NUM_CLASSES; c++)
    Serial.printf("  %s: %d samples\n", myClassLabels[c].c_str(), classCounts[c]);

  if (myTrainingData.empty()) {
    Serial.println("No samples collected yet - Collect for each class first.");
    return;
  }

  std::random_shuffle(myTrainingData.begin(), myTrainingData.end());
  std::vector<TrainingItem> myValData;
  int valCount = 0;
  if (VALIDATION_SAMPLES > 0) {
    int heldOut[NUM_CLASSES] = {};
    std::vector<TrainingItem> trainOnly;
    for (auto& item : myTrainingData) {
      if (heldOut[item.label] < VALIDATION_SAMPLES) {
        myValData.push_back(item); heldOut[item.label]++; valCount++;
      } else {
        trainOnly.push_back(item);
      }
    }
    myTrainingData = trainOnly;
  }
  Serial.printf("Training: %d samples  Validation: %d samples\n", (int)myTrainingData.size(), valCount);

  if (!myAllocateTrainingMemory()) {
    Serial.println("FATAL: training memory alloc failed");
    myFreeTrainingMemory();
    return;
  }

  float* myBatchBuf = (float*)ps_malloc(PACKET_FLOATS * sizeof(float));
  if (!myCheckAlloc(myBatchBuf, "myBatchBuf", PACKET_FLOATS * sizeof(float))) {
    myFreeTrainingMemory();
    return;
  }

  for (int epoch = 0; epoch < TARGET_EPOCHS; epoch++) {
    if (Serial.available() && Serial.peek() == 'x') { Serial.read(); Serial.println("Training stopped by user"); break; }

    std::random_shuffle(myTrainingData.begin(), myTrainingData.end());
    float epochLoss = 0; int correct = 0; int processed = 0;
    myZeroGradients();

    for (int si = 0; si < (int)myTrainingData.size(); si++) {
      if (!myLoadSampleFromFile(myTrainingData[si].path.c_str(), myBatchBuf)) continue;

      myForwardPass(myBatchBuf);
      int label = myTrainingData[si].label;
      epochLoss += myComputeLoss(label);

      int pred = 0;
      for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;
      if (pred == label) correct++;

      myBackwardPass(myBatchBuf, label);
      processed++;

      if ((si + 1) % BATCH_SIZE == 0 || si == (int)myTrainingData.size() - 1) {
        myAdamUpdate(myConv1_w, myConv1_w_grad, myConv1_w_m, myConv1_w_v, CONV1_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myConv1_b, myConv1_b_grad, myConv1_b_m, myConv1_b_v, CONV1_FILTERS, LEARNING_RATE);
        myAdamUpdate(myDense1_w, myDense1_w_grad, myDense1_w_m, myDense1_w_v, DENSE1_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myDense1_b, myDense1_b_grad, myDense1_b_m, myDense1_b_v, DENSE1_SIZE, LEARNING_RATE);
        myAdamUpdate(myDense2_w, myDense2_w_grad, myDense2_w_m, myDense2_w_v, DENSE2_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myDense2_b, myDense2_b_grad, myDense2_b_m, myDense2_b_v, DENSE2_SIZE, LEARNING_RATE);
        myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, OUTPUT_WEIGHTS, LEARNING_RATE);
        myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES, LEARNING_RATE);
        myZeroGradients();
      }
    }

    float avgLoss = processed ? epochLoss / processed : 0;
    float acc = processed ? (100.0f * correct / processed) : 0;
    Serial.printf("Epoch %d/%d  loss=%.4f  acc=%.1f%%\n", epoch + 1, TARGET_EPOCHS, avgLoss, acc);

    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      char buf[24];
      snprintf(buf, sizeof(buf), "Epoch %d/%d", epoch + 1, TARGET_EPOCHS);
      u8g2.drawStr(0, 10, buf);
      snprintf(buf, sizeof(buf), "Acc: %.0f%%", acc);
      u8g2.drawStr(0, 22, buf);
    } while (u8g2.nextPage());
  }

  if (valCount > 0) {
    int correct = 0;
    for (auto& item : myValData) {
      if (!myLoadSampleFromFile(item.path.c_str(), myBatchBuf)) continue;
      myForwardPass(myBatchBuf);
      int pred = 0;
      for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;
      if (pred == item.label) correct++;
    }
    Serial.printf("Validation accuracy: %.1f%% (%d/%d)\n", 100.0f * correct / valCount, correct, valCount);
  }

  free(myBatchBuf);
  myFreeTrainingMemory();
  myWeightsTrained = true;
  mySaveWeights();
  Serial.println("Training complete, weights saved.");
}

void myActionInferOnce() {
  if (!myWeightsTrained) { Serial.println("No trained weights yet - run Train first"); return; }
  if (!myNano33Connected) { Serial.println("Nano33 not connected"); return; }

  if (!myRequestWindowBlocking(MY_CAPTURE_TIMEOUT_MS)) {
    Serial.println("Infer capture timed out");
    return;
  }
  float buf[PACKET_FLOATS];
  memcpy(buf, myPacketFloats, PACKET_BYTES);
  myNormalizePacket(buf);
  myForwardPass(buf);

  int pred = 0;
  for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;

  myPrintResultLine(pred, myFinal_output, myClassLabels, NUM_CLASSES);   // [CHANGED v14] shared format, same as Quick Infer + Nano33
  myServerSendResult(pred, myFinal_output, NUM_CLASSES);   // [NEW v05]

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 10, "Predicted:");
    u8g2.drawStr(0, 22, myClassLabels[pred].c_str());
    char buf2[24];
    snprintf(buf2, sizeof(buf2), "Conf: %.0f%%", myFinal_output[pred] * 100.0f);
    u8g2.drawStr(0, 34, buf2);
  } while (u8g2.nextPage());
}

// Build a full PACKET_FLOATS window by replicating one QUICK reading
// across all WINDOW_TIMESTEPS - trades real temporal motion detail for
// speed. Good enough for a live "is something currently off" read;
// not a substitute for myActionInferOnce()'s real 1-second capture
// when you actually care about motion shape over time.
void myBuildWindowFromQuick(float* outBuf) {
  for (int t = 0; t < WINDOW_TIMESTEPS; t++) {
    memcpy(outBuf + t * FAST_CHANNELS, myQuickFloats, FAST_CHANNELS * sizeof(float));
  }
  memcpy(outBuf + FAST_SIZE, myQuickFloats + FAST_CHANNELS, STATIC_FEATURES * sizeof(float));
}

// Fast continuous-inference reading: one QUICK BLE round trip, replicate
// into a window, forward pass, print the compact scan-friendly format:
//   top: 1normal = 60%, (10%, 60%, 30%)
void myActionQuickInferOnce() {
  if (!myWeightsTrained) { Serial.println("No trained weights yet - run Train first"); return; }
  if (!myNano33Connected) { Serial.println("Nano33 not connected"); return; }

  if (!myRequestQuickBlocking(MY_QUICK_TIMEOUT_MS)) {
    Serial.println("Quick read timed out");
    return;
  }

  float buf[PACKET_FLOATS];
  myBuildWindowFromQuick(buf);
  myNormalizePacket(buf);
  myForwardPass(buf);

  int pred = 0;
  for (int j = 1; j < NUM_CLASSES; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;

  myPrintResultLine(pred, myFinal_output, myClassLabels, NUM_CLASSES);   // [CHANGED v14] shared format, same as Infer + Nano33
  myServerSendResult(pred, myFinal_output, NUM_CLASSES);   // [NEW v05]

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 10, myClassLabels[pred].c_str());
    char buf2[24];
    snprintf(buf2, sizeof(buf2), "%.0f%%", myFinal_output[pred] * 100.0f);
    u8g2.drawStr(0, 22, buf2);
  } while (u8g2.nextPage());
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 7: SERIAL MENU + OLED STATUS                                       ██
// ██████████████████████████████████████████████████████████████████████████████

void myPrintMenu() {
  Serial.println("\n=== Fusion Menu ===");
  for (int i = 0; i < NUM_CLASSES; i++)
    Serial.printf("  %d = Collect '%s'\n", i + 1, myClassLabels[i].c_str());
  Serial.println("  t = Train");
  Serial.println("  i = Infer once/continuous (real 1s window, 'x' to stop)");
  Serial.println("  q = Quick continuous infer (fast, replicated single reading, 'x' to stop)");
  Serial.println("  r = (re)connect to Nano33");
  Serial.println("  k = recalibrate");
  Serial.println("  s = status");
  Serial.println("  p = push current model to Nano33 for on-device inference  [NEW v05]");
  Serial.println("  e = export current model as pasteable hex text over Serial  [NEW v14.2 - BLE-free fallback for the webpage's model fetch]");
  Serial.printf("  d = toggle debug output (currently %s)\n", myDebugVerbose ? "ON" : "OFF");
  Serial.printf("  b = cycle BLE mode (currently %s)  [NEW v07]\n", myBleModeName(myBleMode));
  Serial.println("  z = restart now to apply a newly-selected BLE mode  [NEW v07]");
  Serial.println("  ? = this menu");
}

void myDrawIdleOLED() {
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    char modeBuf[24];
    snprintf(modeBuf, sizeof(modeBuf), "BLE: %s",
             myBleMode == BLE_MODE_DUAL ? "dual" :
             myBleMode == BLE_MODE_PERIPHERAL_ONLY ? "periph" :
             myBleMode == BLE_MODE_CENTRAL_ONLY ? "central" : "off");
    u8g2.drawStr(0, 8, modeBuf);
    u8g2.drawStr(0, 18, myNano33Connected ? "Nano33: linked" : "Nano33: ---");
    if (myHeartbeatFresh) {
      char buf[24];
      snprintf(buf, sizeof(buf), "a:%.1f,%.1f,%.1f", myHbAx, myHbAy, myHbAz);
      u8g2.drawStr(0, 28, buf);
    }
    u8g2.drawStr(0, 38, myWeightsTrained ? "Model: trained" : "Model: none");
  } while (u8g2.nextPage());
}

void myHandleMenuCommand(char cmd) {
  if (cmd >= '1' && cmd <= ('0' + NUM_CLASSES)) {
    myCollectClassIdx = cmd - '1';
    myMode = MODE_COLLECT;
    Serial.printf("\n>>> Collect mode: %s  (press 'c' to capture, 'x' to exit)\n", myClassLabels[myCollectClassIdx].c_str());
    Serial.printf("Current count: %d\n", myCountSamples(myCollectClassIdx));
  } else if (cmd == 't') {
    myActionTrain();
  } else if (cmd == 'i') {
    myMode = MODE_INFER;
    Serial.println("\n>>> Infer mode (press 'x' to exit) - real 1s window capture every ~2s");
  } else if (cmd == 'q') {
    myMode = MODE_QUICK_INFER;
    Serial.println("\n>>> Quick infer mode (press 'x' to exit) - fast replicated reading");
  } else if (cmd == 'r') {
    if (myBleCentralEnabled()) {
      myBLEConnectToNano33();
    } else {
      Serial.printf("Central role disabled in current BLE mode (%s) - press 'b' to change it.\n", myBleModeName(myBleMode));
    }
  } else if (cmd == 'k') {
    myCalibrate();
  } else if (cmd == 'p') {
    myPushModelToNano33();
  } else if (cmd == 'e') {   // [NEW v14.2]
    myExportModelAsHex();
  } else if (cmd == 's') {
    Serial.printf("BLE mode: %s\n", myBleModeName(myBleMode));
    Serial.printf("Nano33 connected: %s\n", myNano33Connected ? "yes" : "no");
    Serial.printf("Browser connected: %s\n", myBrowserConnected ? "yes" : "no");
    Serial.printf("Model trained: %s\n", myWeightsTrained ? "yes" : "no");
    for (int c = 0; c < NUM_CLASSES; c++) Serial.printf("  %s: %d samples\n", myClassLabels[c].c_str(), myCountSamples(c));
  } else if (cmd == 'd') {
    myDebugVerbose = !myDebugVerbose;
    Serial.printf("Debug output now %s\n", myDebugVerbose ? "ON" : "OFF");
  } else if (cmd == 'b') {
    myCycleBleMode();
  } else if (cmd == 'z') {
    Serial.println("Restarting to apply BLE mode...");
    delay(200);
    ESP.restart();
  } else if (cmd == '?') {
    myPrintMenu();
  }
}

void myHandleCollectCommand(char cmd) {
  if (cmd == 'c') {
    myCaptureAndSave(myCollectClassIdx);
    Serial.printf("Count now: %d\n", myCountSamples(myCollectClassIdx));
  } else if (cmd == 'x') {
    myMode = MODE_MENU;
    Serial.println("Back to menu.");
    myPrintMenu();
  }
}

unsigned long myLastInferMs = 0;
void myHandleInferLoop() {
  if (Serial.available() && Serial.peek() == 'x') {
    Serial.read();
    myMode = MODE_MENU;
    Serial.println("Exiting infer mode.");
    myPrintMenu();
    return;
  }
  unsigned long now = millis();
  if (now - myLastInferMs >= 2000) {
    myLastInferMs = now;
    myActionInferOnce();
  }
}

unsigned long myLastQuickInferMs = 0;
void myHandleQuickInferLoop() {
  if (Serial.available() && Serial.peek() == 'x') {
    Serial.read();
    myMode = MODE_MENU;
    Serial.println("Exiting quick infer mode.");
    myPrintMenu();
    return;
  }
  unsigned long now = millis();
  if (now - myLastQuickInferMs >= 300) {   // quick reads are cheap, poll faster
    myLastQuickInferMs = now;
    myActionQuickInferOnce();
  }
}


// ======================================================
// PART 7.5: BROWSER COMMAND DISPATCH [NEW v05] - turns the volatile
// flags set by MyServerControlCallbacks into actual work, called once
// per loop() iteration. Reuses the exact same action functions the
// Serial menu already calls - browser and Serial are just two front
// ends onto the same underlying operations.
// ======================================================
void myHandleBrowserFlags() {
  if (myBrowserWantsCapture) {
    myBrowserWantsCapture = false;
    if (myRequestWindowBlocking(MY_CAPTURE_TIMEOUT_MS)) {
      myServerSendBinary(myPacketBytes, PACKET_BYTES);
    }
  }
  if (myBrowserWantsQuick) {
    myBrowserWantsQuick = false;
    if (myRequestQuickBlocking(MY_QUICK_TIMEOUT_MS)) {
      myServerSendBinary(myQuickBytes, QUICK_PACKET_BYTES);
    }
  }
  if (myBrowserWantsTrain) {
    myBrowserWantsTrain = false;
    myActionTrain();
  }
  if (myBrowserWantsInfer) {
    myBrowserWantsInfer = false;
    myActionInferOnce();          // already calls myServerSendResult() internally
  }
  if (myBrowserWantsQuickInfer) {
    myBrowserWantsQuickInfer = false;
    myActionQuickInferOnce();     // already calls myServerSendResult() internally
  }
  if (myBrowserWantsPushToNano) {
    myBrowserWantsPushToNano = false;
    myPushModelToNano33();
  }
  if (myBrowserWantsPushModel) {
    myBrowserWantsPushModel = false;
    if (!myWeightsTrained) {
      Serial.println("PUSHMODEL requested but no trained weights on this board yet");
    } else {
      myExportModelPackage();
      myServerSendBinary(myModelPackageBuf, MY_MODEL_PACKAGE_BYTES);
      Serial.println("Model package sent to browser over binary channel");
    }
  }
  if (myBrowserWantsReconnect) {
    myBrowserWantsReconnect = false;
    if (myBleCentralEnabled()) {
      myBLEConnectToNano33();
    } else {
      Serial.printf("Reconnect requested but central role disabled in current BLE mode (%s)\n", myBleModeName(myBleMode));
    }
  }
  if (myBrowserWantsCalibrate) {
    myBrowserWantsCalibrate = false;
    myCalibrate();
  }
  if (myBrowserWantsStatus) {
    myBrowserWantsStatus = false;
    Serial.printf("BLE mode: %s   Nano33 connected: %s   Model trained: %s   Browser connected: %s\n",
                  myBleModeName(myBleMode), myNano33Connected ? "yes" : "no", myWeightsTrained ? "yes" : "no",
                  myBrowserConnected ? "yes" : "no");
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 8: SETUP / LOOP                                                    ██
// ██████████████████████████████████████████████████████████████████████████████

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000);
  delay(500);

  Serial.println("\n=== XIAO ESP32-S3 BLE Sensor Fusion System v14 ===");
  Serial.printf("Free heap: %d  Free PSRAM: %d\n", ESP.getFreeHeap(), ESP.getFreePsram());

  myLoadBleModeFromPrefs();   // [NEW v07] restore last-selected BLE mode from NVS
  Serial.printf("BLE mode: %s  (press 'b' to change, 'z' to apply+restart)\n", myBleModeName(myBleMode));

  u8g2.begin();
  u8g2.firstPage();
  do { u8g2.setFont(u8g2_font_5x7_tf); u8g2.drawStr(0, 15, "Starting..."); } while (u8g2.nextPage());

  pinMode(21, OUTPUT);
  digitalWrite(21, HIGH);
  delay(100);
  SPI.begin();
  SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if (!mySDavailable) {
    SD.end();
    Serial.println("WARNING: No SD card detected - Collect/Train/weight-save disabled.");
  } else {
    Serial.println("SD card mounted");
  }

  if (!myAllocateCoreMemory()) {
    Serial.println("FATAL: Core PSRAM allocation failed.");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while (1) delay(1000);
  }

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
  memcpy(myDense1_w, myModel_dense1_w, DENSE1_WEIGHTS * sizeof(float));
  memcpy(myDense1_b, myModel_dense1_b, DENSE1_SIZE    * sizeof(float));
  memcpy(myDense2_w, myModel_dense2_w, DENSE2_WEIGHTS * sizeof(float));
  memcpy(myDense2_b, myModel_dense2_b, DENSE2_SIZE    * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, NUM_CLASSES    * sizeof(float));
  myWeightsTrained = true;
  Serial.println("Baked-in weights loaded");
#endif

  if (myLoadWeights()) Serial.println("SD weights loaded - overriding baked-in weights");

  // ======================================================
  // [CHANGED v07] BLE bring-up now respects myBleMode, and - even in
  // DUAL mode - starts the PERIPHERAL server/advertising BEFORE the
  // blocking central scan runs. In v05/v06 the order was reversed, so
  // the board wasn't advertising at all during the ~5s Nano33 scan +
  // connect + calibration window right after boot; that's the window
  // where a browser's device picker would come up empty. Advertising
  // now starts within ~1s of power-on regardless of Nano33 status.
  // ======================================================
  if (myBleMode != BLE_MODE_OFF) {
    NimBLEDevice::init(MY_ESP32_PERIPHERAL_NAME);
    NimBLEDevice::setMTU(247);

    if (myBlePeripheralEnabled()) {
      myStartBLEPeripheralServer();   // webpage can connect directly to this board
    }
    if (myBleCentralEnabled()) {
      myBLEConnectToNano33();         // best-effort; retry anytime with 'r'
    } else {
      Serial.println("Central role disabled in current BLE mode - skipping Nano33 scan.");
    }
  } else {
    Serial.println("BLE mode OFF - radio not initialized. Press 'b' then 'z' to enable and restart.");
  }

  myCalibrate();   // safe in any mode: loads SD cache if present, else needs an active Nano33 link, else defaults

  myPrintMenu();
  Serial.println("System ready.");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '\n' || cmd == '\r') { /* ignore */ }
    else if (myMode == MODE_MENU)     myHandleMenuCommand(cmd);
    else if (myMode == MODE_COLLECT)  myHandleCollectCommand(cmd);
    // MODE_INFER's and MODE_QUICK_INFER's 'x' is peeked/consumed inside their loop handlers
  }

  if (myMode == MODE_INFER) myHandleInferLoop();
  if (myMode == MODE_QUICK_INFER) myHandleQuickInferLoop();

  myHandleBrowserFlags();   // [NEW v05] act on any pending browser command

  // [NEW v05] relay the Nano33's own heartbeat data on to the browser,
  // so the "ESP32 panel" in index-v02.html shows live sensor values
  // too, not just whichever board's panel happens to be connected.
  static unsigned long lastServerHb = 0;
  if (myBrowserConnected && myServerHeartbeatChar && millis() - lastServerHb >= 500) {
    lastServerHb = millis();
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f,%.2f,%.2f,%.0f,%.1f,%.1f,%d",
             myHbAx, myHbAy, myHbAz, myHbMic, myHbTemp, myHbHumidity, myHbProximity);
    myServerHeartbeatChar->setValue((uint8_t*)buf, strlen(buf));
    myServerHeartbeatChar->notify();
  }

  static unsigned long lastOled = 0;
  if (millis() - lastOled > 500) {
    lastOled = millis();
    if (myMode == MODE_MENU) myDrawIdleOLED();
  }

  static unsigned long lastStatus = 0;
  if (myDebugVerbose && millis() - lastStatus > 3000) {
    lastStatus = millis();
    Serial.printf("[status] connected=%s hbSub=%s binSub=%s lastHeartbeat=%lums ago heartbeatPacketsOK=%s\n",
                  myNano33Connected ? "yes" : "no",
                  myHbSubscribeOk ? "OK" : "FAILED",
                  myBinSubscribeOk ? "OK" : "FAILED",
                  myHeartbeatFresh ? (millis() - myLastHeartbeatMs) : -1,
                  myHeartbeatFresh ? "yes" : "no (none received yet)");
  }
}
