import Darwin
import Foundation

/// Stops a model test whose memory runs away, before it fills the machine's swap.
///
/// A thread reads the process's physical footprint every 100 ms. The footprint counts
/// compressed pages, which the resident size leaves out. Above the limit the guard prints one
/// line, records a one-second stack sample of the process with `/usr/bin/sample`, and exits.
enum MemoryGuard {
  /// The name of the step that is running, for the line the guard prints.
  static let currentStep = Guarded("start")
  private static let started = Guarded(false)

  static func footprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : 0
  }

  static func gigabytes(_ bytes: UInt64) -> String {
    String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
  }

  /// Starts the guard once per process. `prefix` starts every line it prints.
  static func start(limitBytes: UInt64, sampleFile: String?, prefix: String) {
    let first = started.withLock { value -> Bool in
      defer { value = true }
      return !value
    }
    guard first else { return }
    let thread = Thread {
      while true {
        let footprint = footprintBytes()
        if footprint > limitBytes {
          print(
            "\(prefix) ### MEMORY: footprint \(gigabytes(footprint)) > limit \(gigabytes(limitBytes)) "
              + "during \(currentStep.current)")
          fflush(stdout)
          if let sampleFile {
            recordSample(to: sampleFile)
            print("\(prefix) ### MEMORY: stack sample \(sampleFile); footprint now \(gigabytes(footprintBytes()))")
            fflush(stdout)
          }
          _exit(4)
        }
        Thread.sleep(forTimeInterval: 0.1)
      }
    }
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  private static func recordSample(to file: String) {
    let arguments = ["/usr/bin/sample", String(getpid()), "1", "-file", file]
    let cArguments: [UnsafeMutablePointer<CChar>?] = arguments.map { strdup($0) } + [nil]
    defer { cArguments.forEach { free($0) } }
    var child: pid_t = 0
    guard posix_spawn(&child, "/usr/bin/sample", nil, nil, cArguments, environ) == 0 else { return }
    var status: Int32 = 0
    waitpid(child, &status, 0)
  }
}
